import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import CLARA
from config.settings import get_clara_params

def gpu_k_medoids_parallel(cc_set, chunk_models, process_id, gold_data, k, distance,dataset_name):
    """
    Perform parallel k-medoids clustering and score prediction for a chunk of models.
    
    Args:
        cc_set: Pre-selected cc_set indices
        chunk_models: Dictionary of model indices to process
        process_id: Process identifier for GPU allocation
        gold_data: Full dataset of model performances
        k: Number of clusters
        distance: Distance metric for clustering
        max_iter: Maximum iterations for CLARA
        
    Returns:
        Tuple of (process_id, estimated_scores)
    """
    estimated_scores = []
    
    # Determine GPU device
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1
    device = process_id % num_gpus
    
    clara_params = get_clara_params(dataset_name)
    # Process each model in the chunk
    for model in chunk_models:
        # Get gold data for this model's source models
        each_model_golddata = gold_data[chunk_models[model]]
        each_model_golddata_T = each_model_golddata.T
        
        # Standardize data
        scaler = StandardScaler()
        each_model_golddata_T = scaler.fit_transform(each_model_golddata_T)
        
        # Compute distance matrix
        if distance == 'correlation':
            corrs = np.corrcoef(each_model_golddata_T, rowvar=True)
            corrs = 1 - corrs
        else:
            corrs = pairwise_distances(each_model_golddata_T, metric=distance)
        

        
        # Use CLARA clustering
        clara = CLARA(
            metric=distance,
            n_clusters=k,
            init=clara_params['init'],
            n_sampling=clara_params['n_sampling'] or each_model_golddata_T.shape[0],
            n_sampling_iter=clara_params['n_sampling_iter'],
            max_iter=clara_params['max_iter']
        )
        cc_set_array = np.array(cc_set)
        clara.fit(each_model_golddata_T, cc_set_array)
        selected_idxs = clara.medoid_indices_
        
        # Get cluster assignments and sizes
        cluster_members = np.argmin(corrs[selected_idxs, :], axis=0)
        unique, cluster_sizes = np.unique(cluster_members, return_counts=True)
        cluster_sizes = list(cluster_sizes)
        
        # Calculate scores with correction
        means_all_columns = np.mean(each_model_golddata, axis=0)
        means_all_columns = means_all_columns.astype(np.float64) + 0.5
        means_centroids_columns = means_all_columns[selected_idxs]
        observe_centroids_data = gold_data[model][selected_idxs]
        observe_centroids_data = observe_centroids_data.astype(np.float64) + 0.5
        cluster_correction_scale = observe_centroids_data / means_centroids_columns
        
        # Apply correction to clusters
        clustered_indices = [[] for _ in range(k + len(cc_set))]
        for idx, cluster_id in enumerate(cluster_members):
            clustered_indices[cluster_id].append(idx)
        
        for i in range(k + len(cc_set)):
            means_all_columns[clustered_indices[i]] *= cluster_correction_scale[i]
        
        # Finalize scores
        means_all_columns -= 0.5
        means_all_columns[means_all_columns > 1] = 1
        estimated_scores.append(np.mean(means_all_columns))
    
    return (process_id, estimated_scores)