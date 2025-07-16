from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def compute_correlations(true_scores, estimated_scores, corr_fn):
    """
    Compute correlation coefficients and error metrics between true and estimated scores.
    
    Args:
        true_scores: Array of true scores
        estimated_scores: Array of estimated scores
        corr_fn: Dictionary of correlation functions (e.g., {'pearsonr': pearsonr, ...})
        
    Returns:
        Dictionary of computed metrics
    """
    correlations = {}
    for method in corr_fn:
        corr = corr_fn[method](true_scores, estimated_scores).correlation
        correlations[method] = 0 if np.isnan(corr) else corr
    correlations['MAE'] = mean_absolute_error(true_scores, estimated_scores)
    correlations['MSE'] = mean_squared_error(true_scores, estimated_scores)
    return correlations

def acckendall(true_scores, estimated_scores):
    """
    Compute a custom accuracy-based Kendall's tau correlation.
    This is a placeholder implementation assuming it mimics kendalltau.
    
    Args:
        true_scores: Array of true scores
        estimated_scores: Array of estimated scores
    Returns:
        Float representing the correlation
    """
    try:
        # Ensure inputs are numpy arrays
        true_scores = np.array(true_scores)
        estimated_scores = np.array(estimated_scores)
        
        # Compute Kendall's tau as a base correlation
        tau, _ = kendalltau(true_scores, estimated_scores)
        
        # If NaN, return 0 (consistent with error handling in score_cc_set.py)
        if np.isnan(tau):
            return 0.0
        return tau
    except Exception as e:
        print(f"Error in acckendall: {e}")
        return 0.0

def get_sorted_indices_desc(float_list):
    """
    Get indices sorted by value in ascending order (for KL divergence calculation).
    
    Args:
        float_list (list): List of floating-point values
        
    Returns:
        list: Indices sorted by corresponding values in ascending order
    """
    # Use enumerate to get (index, value) pairs
    indexed_list = list(enumerate(float_list))
    # Sort by value ascending (for KL divergence where lower=higher similarity)
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=False)
    # Extract sorted indices
    sorted_indices = [index for index, value in sorted_indexed_list]
    return sorted_indices

def remove_data_from_dict(data_dict, remove_list):
    """
    Remove specified values from all lists in a dictionary.
    
    Args:
        data_dict (dict): Dictionary containing lists as values
        remove_list (list): Values to remove from all lists
        
    Returns:
        dict: Modified dictionary with specified values removed
    """
    # Iterate over each key-value pair
    for key in data_dict:
        # Remove items present in remove_list
        data_dict[key] = [item for item in data_dict[key] if item not in remove_list]
    return data_dict

def compute_correlations(true_scores, estimated_scores, corr_fn):
    """
    Compute correlation metrics between true and estimated scores.
    
    Args:
        true_scores: Array of true scores
        estimated_scores: Array of estimated scores
        corr_fn: Dictionary of correlation functions
        
    Returns:
        Dictionary of correlation values
    """
    correlations = {}
    for method, func in corr_fn.items():
        corr = func(true_scores, estimated_scores).correlation
        correlations[method] = 0 if np.isnan(corr) else corr
    correlations['MAE'] = mean_absolute_error(true_scores, estimated_scores)
    correlations['MSE'] = mean_squared_error(true_scores, estimated_scores)
    return correlations