import json
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations

class EmbeddingCharCrossModel:
    def __init__(self, embs_model1: np.ndarray, embs_model2: np.ndarray):
        
        # model embeddings
        self.A = embs_model1
        self.B = embs_model2
                
    def procrustes_analysis(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, dict]:
        '''
        finds optimal rotation matrix to minimize frobeinus norm dist
        
        returns
            algined A matrix over B space
            
            {initial_sse,
            scaling_factor,
            norm_m2}
        '''
        
        # center the data for better alignment
        A_mean = np.mean(A, axis = 0)
        B_mean = np.mean(B, axis = 0)
        A_c = A - A_mean
        B_c = B - B_mean
        
        # calculate initial sse
        initial_sse = np.sum(np.square(A_c - B_c))
        
        # -- procrustes Analysis -- (kabsch algorithm)
        M = B_c.T @ A_c   # obtain cross-covariance matrix
        U, S, Vt = np.linalg.svd(M) # decompose using SVD
        
        #check for reflection 
        d = np.linalg.det(Vt.T @ U.T)
        if d < 0:
            # flip sign of last element in Vt to force rotation
            Vt[-1, :] *= -1
        
        W = Vt.T @ U.T  # isolate rotation matrix
        
        A_aligned = A_c @ W + B_mean    # compute A aligned to B space
        
        # -- Procrustes Disparity --
        # variances
        var_A = np.sum(np.square(A_c))
        var_B = np.sum(np.square(B_c))
        
        # disparity ((A - B)**2)
        norm_disparity = np.sum(np.square(A_aligned - B_c)) / var_B
        
        # scaling factor
        s = np.sum(S) / var_A
        
        # metrics dict
        metrics = {
            "initial_sse": initial_sse,
            "scaling_factor": s,
            "norm_m2": norm_disparity,
        }
        
        return A_aligned, metrics
    
    def pairwise_similarities(self, A: np.ndarray, B: np.ndarray):
        '''cosine similarity between equivalent embeddings'''
        
        # use the Hadamard product (element-wise mulitplication)
        similarities = np.sum(A * B, axis = 1) # similarity vector (N, 1)
        
        return similarities.mean(), similarities.std()
    
    def random_similarities(self, A: np.ndarray, B: np.ndarray):
        '''cosine similarities between random pairs'''
        
        # shuffle B
        rng = np.random.default_rng()
        B_shuffled = rng.permutation(B, axis=0)
        
        # obtain similarities
        random_sims = np.sum(A * B_shuffled, axis = 1)
        
        return random_sims.mean(), random_sims.std()
    
    def precision_at_k(self, sim_matrix: np.ndarray, k_list: list[int] = [1, 5, 10]):
        '''returns precision @ k scores from a similarity matrix'''
        
        # get the indices of the top k closest embeddings per row
        top_k_idx = np.argpartition(-sim_matrix, max(k_list), axis = 1)[:, :max(k_list)]
        
        # target indices (diagonal elements of the matrix)
        targets = np.argparse(sim_matrix.shape[0]).reshape(-1, 1)   # [[0], [1], [2], [3] ...]
        
        results = {}
        for k in k_list:
            # check if there are mathces in the first k closest elements
            matches = np.any(top_k_idx[:, :k] == targets, axis = 1)
            results[f"P@{k}"] = matches.mean()  # fraction of matches in topk elements
            
        return results
        
    def csls_matrix(self, A: np.ndarray, B: np.ndarray, k_neighbors: int = 10):
        '''computes similarity matrix accounting for hubness correction'''
        
        # compute similarity matrix
        sim_matrix = A @ B.T
        
        # mean similarity of each source to it's k nearest neighbors
        r_S = np.partition(sim_matrix, -k_neighbors, axis=1)[:, -k_neighbors:].mean(axis=1)
        
        # mean similarity of each target to it's k nearest sources
        r_T = np.partition(sim_matrix, -k_neighbors, axis=0)[-k_neighbors:, :].mean(axis=0)
        
        # csls matrix (as per standard forumla, considering that A and B are norm)
        csls = 2 * sim_matrix - r_S[:, None] - r_T[None, :]
        
        return csls
        
        
    def compute_all_metrics(self):
        '''compute all metrics and return all in json format'''

        # copy matrice objects
        A = self.A.copy()
        B = self.B.copy()
        
        # align matrix A to sapce B
        A_aligned, procrustes_results = self.procrustes_analysis(A, B)
        
        # obtain pair-wise and non-pair-wise similarities
        pw_sim_mean, pw_sim_std = self.pairwise_similarities(A_aligned, B)
        npw_sim_mean, npw_sim_std = self.random_similarities(A_aligned, B)
        
        # get precision at k = [1, 5, 10]
        precision_results = self.precision_at_k(A_aligned @ B.T, k_list = [1, 5, 10])
        
        # obtain csls matrix
        csls_sim_matrix = self.csls_matrix(A_aligned, B, k_neighbors = 10)
        
        # new precision scores with csls matrix
        csls_precision_results = self.precision_at_k(csls_sim_matrix, k_list = [1, 5, 10])
        
        # save results as json
        results = {
            'procrustes_results': procrustes_results,
            'pair-wise-sim': {
                'mean': pw_sim_mean,
                'std': pw_sim_std,
            },
            'random-sim': {
                'mean': npw_sim_mean,
                'std': npw_sim_std,
            },
            'precision_results': precision_results,
            'csls_precision_results': csls_precision_results,
        }
        
        return results
        
# internal function to ease use
def extract_embeddings(npz_path: Path):
    '''extract embeddings and corresponding file_names'''
    
    # load data
    data = np.load(npz_path)
    
    # extract names and embeddings
    train_fnames = data['train_filenames']
    train_embs = data['train_embeddings']
    test_fnames = data['test_filenames']
    test_embs = data['test_embeddings']

    return train_fnames, train_embs, test_fnames, test_embs

        
if __name__ == "__main__":
    
    # dict with model and lang
    model_lang = {
        "model1": "es",
        "model2": "en",
        "model3": "it",
        "model4": "es",
    }
    
    # paths
    results_path = Path("/gpfs/projects/bsc14/mouhida/TFG/results/cross_model") 
    results_path.mkdir(parents = True, exist_ok = True)
    embs_path = Path("/gpfs/projects/bsc14/mouhida/TFG/embeddings")
    
    # iter through all model combinations
    for model_1, model_2 in combinations(list(model_lang), 2):
        
        # dont compare same language models
        if model_lang[model_1] == model_lang[model_2]:
            continue

        print(f"[INFO] Comparing models: {model_1} - {model_2}")

        # paths for both models
        model_1_path = embs_path / model_1
        model_2_path = embs_path / model_2
        
        langs = [model_lang[model_1], model_lang[model_2]]

        for lang in langs:
            
            # start timing
            start_time = datetime.now()

            # obtain embs for each model in that lang
            _, emb_model_1, _, _ = extract_embeddings(model_1_path / f"emb_{lang}.npz")
            _, emb_model_2, _, _ = extract_embeddings(model_2_path / f"emb_{lang}.npz")

            # obtain metrics
            pipeline = EmbeddingCharCrossModel(emb_model_1, emb_model_2)
            char_results = pipeline.compute_all_metrics()

            # append model characteristics
            char_params = {
                'model_1': model_1,
                'model_2': model_2,
                'lang': lang
            }

            results = char_params | char_results

            # save results
            lang_path = results_path / lang
            lang_path.mkdir(parents = False, exist_ok = True)

            with open(lang_path / f"results_{model_1}_{model_2}.json", "w") as f:
                json.dump(results, f, indent = 4)

            # calculate total time
            end_time = datetime.now()
            elapsed = end_time - start_time
            
            print(f"\t[INFO] Completed analysis for language {lang} in {elapsed}")