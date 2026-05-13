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
        self.emb_path = Path("/gpfs/projects/bsc14/mouhida/TFG/embeddings")
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
        
    def embed_train_texts(self):
        '''embed files in the train directory'''
    
        files = sorted(self.train_path.rglob("txt/*.txt"))
        names = [f.stem for f in files]
        
        train_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(files), self.batch_size)):
                batch_files = files[i:i + self.batch_size]
                batch = [f.read_text(encoding = "utf-8") for f in batch_files]
                enc = self.tokenizer(
                    batch, padding=True, 
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                out = self.model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                train_embs.append(emb.cpu().numpy())
                
        embeddings = np.concatenate(train_embs, axis = 0)
        
        return names, embeddings
        
    def embed_test_texts(self):
        '''embed the files in the test directory that belong to the test set'''

        # paths for test files mapping
        self.test_map_path = self.test_path / "MultiClinAI_filenames_mapping_test.tsv"

        # obtain list of files that belong to the test set (not background)
        map_df = pd.read_csv(self.test_map_path, sep="\t")
        test_task_fnames = map_df.loc[map_df['test_or_background'] == "test", "task_fname"].to_list()
        test_stems = {Path(name).stem for name in test_task_fnames}

        # keep only txt files that belong to the test set
        all_files = sorted(self.test_path.rglob("txt/*.txt"))
        files = [f for f in all_files if f.stem in test_stems]
        names = [f.stem for f in files]
        
        #XXX: theoretically i can just take the list of files from the files map

        test_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(files), self.batch_size)):
                batch_files = files[i:i + self.batch_size]
                batch = [f.read_text(encoding = "utf-8") for f in batch_files]
                enc = self.tokenizer(
                    batch, padding = True, 
                    truncation = True,
                    return_tensors = "pt"
                ).to(self.device)
                out = self.model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                test_embs.append(emb.cpu().numpy())

        embeddings = np.concatenate(test_embs, axis=0)

        return names, embeddings
    
    def embed_files(self):
        '''embed all files and save as .npz'''
        
        # time how long it takes
        start_time = datetime.now()
        
        print("\t[INFO] Embedding Train Files")
        train_names, train_embs = self.embed_train_texts()
        print("\t[INFO] Embedding Test Files")
        test_names, test_embs = self.embed_test_texts()
        
        out_path = self.emb_path / f"{self.model_name}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            train_filenames=np.array(train_names),
            train_embeddings=train_embs,
            test_filenames=np.array(test_names),
            test_embeddings=test_embs,
        )
        
        # timing
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\t[SUCESS] Embedded all files in {duration}")
        
if __name__ == "__main__":
    
    # obtain model name as argument from the run file
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help = "model to be used for the embedding")
    args = parser.parse_args()
    
    embedding_processor = EmbeddingProcessor(args.model)
    embedding_processor.embed_files()