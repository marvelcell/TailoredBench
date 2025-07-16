import joblib
from models.prediction import gpu_k_medoids_parallel


def worker(args_tuple):
    """Wrapper function to ensure proper argument passing to gpu_k_medoids_parallel"""

    cc_set, chunk_models, pid, gold_data, k, distance, dataset_name = args_tuple
    return gpu_k_medoids_parallel(cc_set, chunk_models, pid, gold_data, k, distance, dataset_name)


def split_dict(d, num_splits):
    """
    Split dictionary into approximately equal-sized sub-dictionaries.
    
    Args:
        d (dict): Input dictionary to split
        num_splits (int): Number of sub-dictionaries to create
        
    Returns:
        list: List of num_splits dictionaries
        
    Raises:
        ValueError: If num_splits <= 0
    """
    if num_splits <= 0:
        raise ValueError("Number of splits must be greater than 0")
    total_items = len(d)
    part_size = total_items // num_splits
    remainder = total_items % num_splits
    split_dicts = [{} for _ in range(num_splits)]
    items = list(d.items())
    start = 0
    for i in range(num_splits):
        end = start + part_size + (1 if i < remainder else 0)
        for j in range(start, end):
            split_dicts[i][items[j][0]] = items[j][1]
        start = end
    return split_dicts