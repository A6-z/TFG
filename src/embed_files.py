import re
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from transformers import AutoModel, AutoTokenizer

class EmbeddingProcessor:
    def __init__(self, model_name):
        
        # define model name and path
        self.model_name = model_name
        self.model_path = Path("/gpfs/projects/bsc14/mouhida/models/tfg") / self.model_name
        
        # files paths
        self.data_path = Path('/gpfs/projects/bsc14/mouhida/datasets/multiclinai')
        self.train_path = self.data_path / "train/MultiClinNER"  # train files path
        self.test_path = self.data_path / "test"    # test files path
        
        # path to save embeddings
        self.emb_path = Path("/gpfs/projects/bsc14/mouhida/TFG/embeddings") / self.model_name
        self.emb_path.mkdir(parents = True, exist_ok = True)
        
        # load tokenizer
        try:
            # try fast tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        except ValueError:
            # fallback to slow tokenizer if fast is broken/unavailable for that model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        
        # load model
        self.model = AutoModel.from_pretrained(self.model_path)
        
        # set device and move the model        
        assert torch.cuda.is_available(), "CUDA not available — aborting"
        self.device = "cuda"
        self.model.to(self.device).eval()
        
        # batch size and max len
        self.batch_size = 64
        self.max_length = 512
        
        # set languages
        self.langs = ["cz", "en", "es", "it", "nl", "ro", "sv"]
        
    def alignment_sort_key(self, file_path: Path):
        '''ensures that files are aligned by category and numbering across languages'''
        
        stem = file_path.stem # extract stem
        
        match = re.search(r'-([a-z]+)-(\d+)$', stem)
        
        if match:
            category = match.group(1)
            index = int(match.group(2))
            return (category, index)
        
        return ("", 0)  # fallback in case of no match
    
    def embed_texts(self, files_list: list):
        '''takes a list of files and returns the sorted names and embeddings'''
        
        # sort files and pull sorted names
        files_list.sort(key = self.alignment_sort_key)
        names = [f.stem for f in files_list]
        
        embs = [] # list to store embeddings
        
        # embed files
        with torch.no_grad():
            for i in tqdm(range(0, len(files_list), self.batch_size)):
                batch_files = files_list[i:i + self.batch_size]
                batch = [f.read_text(encoding = "utf-8") for f in batch_files]
                enc = self.tokenizer(
                    batch,
                    padding = True,
                    truncation = True,
                    return_tensors = "pt"
                ).to(self.device)
                out = self.model(**enc)
                mask = enc['attention_mask'].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min = 1e-9) # mean pool embeddings 
                embs.append(emb.cpu().numpy())
            
        # turn int matrix
        embeddings = np.concatenate(embs, axis = 0)

        return names, embeddings
        
    def embed_files(self):
        '''embed train and test files'''

        # time the process
        start_time = datetime.now()

        # TRAIN FILES
        # obtain all .txt files from the train set
        all_train_files = sorted(self.train_path.rglob("txt/*.txt"))

        # TEST FILES
        # get file of names belonging to the test set
        test_files_map = pd.read_csv(self.test_path / "MultiClinAI_filenames_mapping_test.tsv", sep="\t")
        test_set_fnames = test_files_map.loc[test_files_map['test_or_background'] == "test", "task_fname"].to_list()
        test_stems = {Path(name).stem for name in test_set_fnames} # set of test fnames

        # get .txt files from test directory
        bloated_test_files = sorted(self.test_path.rglob("txt/*.txt"))
        all_test_files = [f for f in bloated_test_files if f.stem in test_stems]

        # iter through languages
        for lang in self.langs:
            
            lang_pattern = f"-{lang}-"  # pattern

            # EMBED TRAIN FILES
            print(f"\t[INFO] Embedding Train Files for {lang}")
            train_files_lang = [f for f in all_train_files if lang_pattern in str(f)] # filter train files
            train_names, train_embs = self.embed_texts(train_files_lang)

            # EMBED TEST FILES
            print(f"\t[INFO] Embedding Test Files for {lang}")
            test_files_lang = [f for f in all_test_files if lang_pattern in str(f)] # filter test files
            test_names, test_embs = self.embed_texts(test_files_lang)

            # save as .npz
            out_path = self.emb_path / f"emb_{lang}.npz" # dir to save embeddings

            np.savez(
                out_path,
                train_filenames = np.array(train_names),
                train_embeddings = train_embs,
                test_filenames = np.array(test_names),
                test_embeddings = test_embs,
            )
            
        # timing
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\t[SUCCESS] Embedded all files in {duration}")

if __name__ == "__main__":
    
    # obtain model name as argument from the run file
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help = "model to be used for the embedding")
    args = parser.parse_args()
    
    embedding_processor = EmbeddingProcessor(args.model)
    embedding_processor.embed_files()
