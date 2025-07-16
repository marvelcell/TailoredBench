import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import CLARA
from config.settings import get_clara_params

def score_cc_set_selection(source_models_golddata, target_models_golddata, num_cc_set, distance, dataset_name):
    """
    Select cc_set using CLARA clustering based on source model data.
    
    Args:
        source_models_golddata: Source model performance data
        target_models_golddata: Target model performance data (unused here but kept for consistency)
        num_cc_set: Number of clusters (cc_set size)
        distance: Distance metric for clustering
        
    Returns:
        Sorted list of selected indices (cc_set)
    """
    # Transpose source data so each row is an example
    source_models_golddata = source_models_golddata.T
    
    # Standardize data
    scaler = StandardScaler()
    source_models_golddata = scaler.fit_transform(source_models_golddata)
    # Get CLARA parameters based on dataset type
    clara_params = get_clara_params(dataset_name)
    
    # Use CLARA clustering
    clara = CLARA(
        metric=distance,
        n_clusters=num_cc_set,
        init=clara_params['init'],
        n_sampling=source_models_golddata.shape[0],
        n_sampling_iter=1,
        max_iter=5000
    )
    clara.fit(source_models_golddata)
    
    # Get selected indices and sort them
    selected_idxs = clara.medoid_indices_
    cc_set = list(selected_idxs)
    cc_set.sort()
    
    return cc_set

def get_target_model_corrs_models(corrs, source_models, target_models):
    """
    Calculate average number of associated source models per target model based on correlation similarity.
    
    Args:
        corrs: Pairwise correlation matrix between models (n_models x n_models)
        source_models: List of source model indices
        target_models: List of target model indices
        
    Returns:
        avg_num: Average count of source models associated with each target model
    """
    n_models = corrs.shape[0]
    upper_tri_indices = np.triu_indices(n_models, k=1)
    distances = corrs[upper_tri_indices]
    model_i_indices = upper_tri_indices[0]  
    model_j_indices = upper_tri_indices[1]  
    
    pairs = list(zip(distances, model_i_indices, model_j_indices))  
    
    pairs.sort(key=lambda x: x[0])
    
    total_similarity = 0
    num_pairs = 0
    for each_pair in pairs:
        num_pairs += 1
        total_similarity += each_pair[0]

    quantile_similarity = total_similarity/num_pairs   

    pairs_in_p_percent = [tup for tup in pairs if tup[0] <= quantile_similarity]
    cutoff_index = len(pairs_in_p_percent)

    target_model_corrs_models = {}
    for t in target_models:
        source_list = []
        added_sources = set()

        for distance, i, j in pairs_in_p_percent:
            if t == i and j in source_models and j not in added_sources:
                source_list.append(j)
                added_sources.add(j)
            elif t == j and i in source_models and i not in added_sources:
                source_list.append(i)
                added_sources.add(i)
        target_model_corrs_models[t] = source_list
    
    num = 0
    all_length = 0
    for item in target_model_corrs_models:
        num += 1
        all_length += len(target_model_corrs_models[item])
    all_length = all_length / num
    return int(all_length)
