import pandas as pd
from pathlib import Path

# file paths
train_files_path = Path('/gpfs/projects/bsc14/mouhida/datasets/multiclinai/train')
test_files_path = Path('/gpfs/projects/bsc14/mouhida/datasets/multiclinai/test')

# get non-background set fnames from test set
test_map_path = test_files_path / 'MultiClinAI_filenames_mapping_test.tsv'  # mappings file
map_df = pd.read_csv(test_map_path, sep = "\t")
test_task_fnames = map_df[map_df['test_or_background'] == "test", "task_fname"].to_list()


class EmbeddingProcessor:
    def __init__(self, model_path):
        
        self.model = 

