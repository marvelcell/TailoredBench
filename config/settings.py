
import os
from typing import Dict, Any

# Dataset configurations
DATASETS = {
    'gsm8k': {
        'source_data_path': './BenchmarkDatasets/gsm8k/source_model_alldata_acc_0.npy',
        'target_data_path': './BenchmarkDatasets/gsm8k/target_model_alldata_acc_0.npy',
        'score_type': 'accuracy',
        'requires_gt': False
    },
    'winogrande': {
        'source_data_path': './BenchmarkDatasets/winogrande/source_model_alldata_acc_0.npy',
        'target_data_path': './BenchmarkDatasets/winogrande/target_model_alldata_acc_0.npy',
        'score_type': 'accuracy',
        'requires_gt': False
    },
    'POPE': {
        'source_data_path': './BenchmarkDatasets/POPE/source_model_alldata_acc_0.npy',
        'target_data_path': './BenchmarkDatasets/POPE/target_model_alldata_acc_0.npy',
        'score_type': 'accuracy',
        'requires_gt': False
    },
    'arc_challenge': {
        'source_data_path': './BenchmarkDatasets/arc_challenge/source_model_alldata.npy',
        'target_data_path': './BenchmarkDatasets/arc_challenge/target_model_alldata.npy',
        'gt_path': './BenchmarkDatasets/arc_challenge/gt_label.npy',
        'score_type': 'accuracy',
        'requires_gt': True
    },
    'hellaswag': {
        'source_data_path': './BenchmarkDatasets/hellaswag/source_model_alldata.npy',
        'target_data_path': './BenchmarkDatasets/hellaswag/target_model_alldata.npy',
        'gt_path': './BenchmarkDatasets/hellaswag/gt_label_acl.npy',
        'score_type': 'accuracy',
        'requires_gt': True
    }
}

# Model configurations
MODEL_CONFIG = {
    'num_source_models': 69,     
    'max_dataset_size': 6000,
    'max_samples': 6000
}

# Clustering configurations
CLUSTERING_CONFIG = {
    'clara_params': {
        'standard': {
            'n_sampling': None,  # Will be set to data shape
            'n_sampling_iter': 1,
            'max_iter': 9000,
            'init': 'random'
        },
        'arc_hella': {
            'n_sampling': 750,
            'n_sampling_iter': 10,
            'max_iter': 5000,
            'init': 'random'
        }
    },
    'distances': ['manhattan', 'euclidean', 'cosine', 'correlation', 'braycurtis'],
    'default_distance': 'manhattan',
    'num_cc_set': 10,
    'point_counts': [20]
}

# Correlation metrics
CORRELATION_METRICS = ['pearsonr', 'kendalltau', 'spearmanr']

# Evaluation metrics
EVALUATION_METRICS = ['pearsonr', 'kendalltau', 'spearmanr', 'MAE', 'MSE']

# Parallel processing configurations
PARALLEL_CONFIG = {
    'num_processes': 24,
    'num_runs': 100,
    'use_gpu': True,
    'gpu_device_selection': 'auto'  # 'auto' or specific device id
}

# Experiment configurations
EXPERIMENT_CONFIG = {
    'experiments': [
        'within_each_family_glue',
        'transfer_across_families_glue',
        'openai_heldout_glue',
        'mmlu',
        'domain_cc_set',
        'score_cc_set'
    ],
    'default_experiment': 'score_cc_set'
}

# File paths and directories
PATHS = {
    'data_root': './BenchmarkDatasets',
    'results_root': './results',
    'anchor_points_root': './AnchorPoints'
}

# Output configurations
OUTPUT_CONFIG = {
    'excel_sheets': ['pearsonr', 'kendalltau', 'spearmanr', 'MAE', 'MSE'],
    'precision': {
        'correlation': 4,
        'error': 6
    },
    'file_formats': {
        'results': 'txt',
        'detailed_results': 'xlsx',
        'scores': 'npy'
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file']
}

# Data preprocessing
PREPROCESSING_CONFIG = {
    'standardization': True,
    'score_adjustment': {
        'alpaca_eval': -1,  # Subtract 1 from average preference
        'other_datasets': 0
    },
    'threshold': 1e-8,  # For GSM8K-like datasets
    'score_bounds': {
        'min': 0,
        'max': 1
    }
}

# Validation and error checking
VALIDATION_CONFIG = {
    'check_file_existence': True,
    'validate_data_shapes': True,
    'handle_nan_values': True,
    'min_samples_per_cluster': 1
}

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in configuration")
    return DATASETS[dataset_name]

def get_clara_params(dataset_name: str) -> Dict[str, Any]:
    if dataset_name in ['arc_challenge', 'hellaswag']:
        return CLUSTERING_CONFIG['clara_params']['arc_hella']
    return CLUSTERING_CONFIG['clara_params']['standard']

def get_output_directory(dataset_name: str, experiment_type: str = 'default') -> str:
    """Generate output directory path."""
    base_path = f"{PATHS['results_root']}/{dataset_name}_result"
    if experiment_type == 'score_cc_set':
        return f"{base_path}/clusterunionset_quantile"
    return base_path

def get_anchor_points_directory(dataset_name: str) -> str:
    """Generate anchor points directory path."""
    return f"{PATHS['anchor_points_root']}/{dataset_name}_result/clusterunionset_quantile"

# Environment variables
def setup_environment():
    """Setup environment variables."""
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    
    # Set number of threads for numpy operations
    os.environ['OMP_NUM_THREADS'] = str(PARALLEL_CONFIG['num_processes'])
    os.environ['MKL_NUM_THREADS'] = str(PARALLEL_CONFIG['num_processes'])

# Default argument configurations for backward compatibility
DEFAULT_ARGS = {
    'dataset': 'all',
    'root_dir': '',
    'results_dir': '',
    'corr_type': ['pearsonr', 'kendalltau', 'spearmanr'],
    'max_samples': 6000,
    'datasets_to_run': ['gsm8k'],
    'max_dataset_size': 6000,
    'num_source_models': 69,
    'num_runs': 100,
    'point_counts': [20],
    'exp': 'score_cc_set',
    'num_cc_set': 10,
    'num_processes': 24,
    'distance': 'manhattan'
}

# Utility functions
def is_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_num_gpus() -> int:
    """Get number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0