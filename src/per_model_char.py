import json
import umap
import hdbscan
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class EmbeddingCharPerSpace:
    '''outputs characteristics of a specific embedding space as a json file'''
    def __init__(self, embeddings: np.ndarray):

        # ensure L2 normalization 
        norms = np.linalg.norm(embeddings, axis = 1)
        if not np.allclose(norms, 1.0, atol = 1e-5):
            raise ValueError(
                f"Embeddings must be L2-normalized; got norms in "
                f"[{norms.min():.6f}, {norms.max():.6f}]"
            )

        self.embeddings = embeddings
        self.glob_sim = np.dot(embeddings, embeddings.T) # just dot product bc they are norm embs

    def effective_dimensions(self, embeddings, var_perc : int = 90):
        '''compute number of dimensions needed to explaini x% of the total variance'''
       
        pca = PCA()  # instantiate pca module
        pca.fit(embeddings)  # fit the emebddings
        cumvar = np.cumsum(pca.explained_variance_ratio_) # compute cumvar explained across comps
        eff_dims = np.searchsorted(cumvar, var_perc / 100) + 1  # record num comps needed for var_perc

        return int(eff_dims)
        # higher dims = more geometric spread
   
    def cos_sim_dist(self, embeddings):
        '''compute the mean and std of the cos similarities in the emb space (ISOTROPY)'''

        # to calculate mean and std we just use top triangle (avoid repetitions)
        iu = np.triu_indices(self.glob_sim.shape[0], k = 1)
        off_diag_values = self.glob_sim[iu]
        
        return off_diag_values.mean(), off_diag_values.std()
        # high mean = anisotropy
        
    def clustering_behaviour(self, embeddings, dimensions: int = 15):
        '''clustering using UMAP + HDBSCAN'''

        # -- UMAP REDUCTION--
        
        # umap config
        UMAP_CONFIG = {
            "n_components": dimensions,     # number of dims to reduce to
            "n_neighbors": int(np.clip(np.sqrt(len(embeddings)), 10, 50)),     # local neighbourhood size
            "min_dist": 0.0,         # tight packing = better clustering
            "metric": "cosine",     # like the similarity metric
            "random_state": 42,     # reproducibility   
        }
        
        # reduce dimentions
        reducer = umap.UMAP(**UMAP_CONFIG)
        emb_reduced = reducer.fit_transform(embeddings)     # (N, dimensions)
        
        # -- HDBSCAN CLUSTERING --
        
        # hdbscan config
        HDBSCAN_CONFIG = {
            "min_cluster_size": 10,     
            "min_samples": 5,           # higher = less noise
            "metric": "euclidean",       # standard with umap
            "cluster_selection_method": "eom",       # exess of mass
            "prediction_data": True,        # to extract membership and persistence
        }
        
        # cluster
        clusterer = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)
        clusterer.fit(emb_reduced)
        
        # extract labels
        labels = clusterer.labels_      # noise samples are labeled -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # substract 1 if there's noise
        
        # quantify noise
        noise_mask = labels == -1       # get noise sample points
        noise_fraction = noise_mask.sum() / len(embeddings)

        # persistence scores
        persistence_per_node = clusterer.cluster_persistence_.tolist()
        mean_persistence = np.mean(persistence_per_node)
        std_persistence = np.std(persistence_per_node)
        
        # -- PER CLUSTER METRICS -- 
        per_cluster_sims = []
        
        for cluster_id in range(n_clusters):
            mask = (labels == cluster_id)
            cluster_size = int(mask.sum())

            if cluster_size < 2:
                intra_cosine = 1.0
            else:
                cluster_similarities = self.glob_sim[mask][:, mask]
                
                # sum upper triangle of similarities (all - diagonal) / 2
                sum_upper = (cluster_similarities.sum() - np.trace(cluster_similarities)) / 2
                num_pairs = (cluster_size * (cluster_size - 1 )) / 2    # num of pairs
                intra_cosine = float(sum_upper / num_pairs) # mean cosine sim

            per_cluster_sims.append(intra_cosine)

        # calculate mean and std of intracluster sims
        mean_cluster_sim = np.mean(per_cluster_sims)
        std_cluster_sim = np.std(per_cluster_sims)

        return n_clusters, noise_fraction, mean_persistence, std_persistence, mean_cluster_sim, std_cluster_sim
    
    def local_intrinsic_dimensionality(self, embeddings, k_neighbors: int = 20):
        '''the dimensions of the local maniforl where the data lives'''

        # obtain distances to neighbors
        nbrs = NearestNeighbors(n_neighbors = k_neighbors + 1, metric = 'euclidean').fit(embeddings) # k + 1 to not count self
        distances, _ = nbrs.kneighbors(embeddings)
        dists = distances[:, 1:] # exclude self

        # -- LID -- 
        dists = np.maximum(dists, 1e-12) # prevent underflow
        r_k = dists[:, -1:] # extract max local dist
        
        # obtain ration 
        ratios = dists[:, :-1] / r_k    # exclude last column to avoid log(1) = 0
        log_ratios = np.log(ratios) # compute log
        
        # calculate dim and guard againts small dists
        mean_log = np.mean(log_ratios, axis = 1) # over k - 1 elements

        # discard degenerate points (duplicates / collinear neighborhoods)
        valid = mean_log < 0
        
        # fallback if the entire dataset is degenerate duplicates
        if not np.any(valid):
            return float('nan')

        lid_scores = -1.0 / mean_log[valid]

        # median is more robust than mean for the heavy-tailed LID distribution
        return float(np.median(lid_scores))

    def characterize_embs(self):
        '''calculate all emb characteristics and return in json format'''
        
        # obtain effective dimensions for 90% of variance
        effective_dimensions = self.effective_dimensions(self.embeddings)

        # get cosine similarity distribution
        mean_cos_sim, std_cos_sim = self.cos_sim_dist(self.embeddings)

        # characterize clusters (umap dims = 15)
        (n_clusters,
         noise_fraction,
         mean_persistence,
         std_persistence,
         mean_cluster_sim,
         std_cluster_sim) = self.clustering_behaviour(self.embeddings)
        
        # obtain mean LID (k = 20)
        mean_lid = self.local_intrinsic_dimensionality(self.embeddings, k_neighbors = 20)

        # save everything in json format:
        results = {
            "effective_dimensions": int(effective_dimensions),
            "cosine_similarity": {
                "mean": float(mean_cos_sim),
                "std": float(std_cos_sim),
            },
            "number_of_clusters": int(n_clusters),
            "noise_fraction": float(noise_fraction),
            "persistence": {
                "mean": float(mean_persistence),
                "std": float(std_persistence)
            },
            "intercluster_cosine_similarity": {
                "mean": float(mean_cluster_sim),
                "std": float(std_cluster_sim),
            },
            "local_intrinsic_dimensionality": float(mean_lid),
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
   
    # paths
    embs_path = Path("/gpfs/projects/bsc14/mouhida/TFG/embeddings")
    results_path = Path("/gpfs/projects/bsc14/mouhida/TFG/results/per_model")
    results_path.mkdir(parents = True, exist_ok = True)
   
    for model_path in embs_path.iterdir():
        
        model_name = model_path.stem # get lang
        print(f"[INFO] Starting characterization for model {model_name}")
        
        for embedding_dir in model_path.iterdir():

            # start timing
            start_time = datetime.now()
            
            lang = str(embedding_dir.stem)[-2:] # get lang

            # extract embeddings
            _, train_embs, _, test_embs = extract_embeddings(embedding_dir)

            # use both train adn test embeddings
            embeddings = np.concatenate([train_embs, test_embs], axis = 0)

            # extract metrics
            pipeline = EmbeddingCharPerSpace(embeddings)
            char_results = pipeline.characterize_embs() 

            # model parameters to save into final json
            model_params = {
                "model": model_name,
                "lang": lang,
                "embeddings_dims": embeddings.shape[-1]
            }
            
            # create a merged dict
            results = model_params | char_results

            # out dir
            output_path = results_path / model_name 
            output_path.mkdir(parents = False, exist_ok = True)
            
            # save results as json
            with open(output_path / f"results_{lang}.json", "w") as f:
                json.dump(results, f, indent = 4)
                
            # calculate total time
            end_time = datetime.now()
            elapsed = end_time - start_time
            
            print(f"\t[INFO] Completed analysis for language {lang} in {elapsed}")
            