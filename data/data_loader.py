"""
Data loading utilities for model performance predictor.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from config import get_dataset_config, VALIDATION_CONFIG


class ModelDataLoader:
    """Handles loading and validation of model data."""
    
    def __init__(self, validate_data: bool = True):
        """
        Initialize the data loader.
        
        Args:
            validate_data: Whether to validate loaded data
        """
        self.validate_data = validate_data
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load source and target model data for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (source_models_data, target_models_data)
            
        Raises:
            FileNotFoundError: If data files don't exist
            ValueError: If dataset is not configured
        """
        try:
            config = get_dataset_config(dataset_name)
            
            source_path = config['source_data_path']
            target_path = config['target_data_path']
            
            # Check if files exist
            if self.validate_data and VALIDATION_CONFIG['check_file_existence']:
                self._check_file_existence(source_path, target_path)
            
            # Load data
            self.logger.info(f"Loading source data from: {source_path}")
            source_data = np.load(source_path, allow_pickle=True)
            
            self.logger.info(f"Loading target data from: {target_path}")
            target_data = np.load(target_path, allow_pickle=True)
            
            self.logger.info(f"Successfully loaded {dataset_name} dataset")
            self.logger.info(f"Source data shape: {source_data.shape}")
            self.logger.info(f"Target data shape: {target_data.shape}")
            
            return source_data, target_data
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            raise
    
    def load_multiple_datasets(self, dataset_names: list) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load multiple datasets.
        
        Args:
            dataset_names: List of dataset names to load
            
        Returns:
            Dictionary mapping dataset names to (source_data, target_data) tuples
        """
        datasets = {}
        
        for dataset_name in dataset_names:
            try:
                datasets[dataset_name] = self.load_dataset(dataset_name)
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
                # Continue loading other datasets
                continue
        
        return datasets
    
    def _check_file_existence(self, source_path: str, target_path: str) -> None:
        """Check if data files exist."""
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source data file not found: {source_path}")
        
        if not Path(target_path).exists():
            raise FileNotFoundError(f"Target data file not found: {target_path}")
        

    @staticmethod
    def get_data_info(data: np.ndarray) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary containing data information
        """
        info = {
            'shape': data.shape,
            'dtype': data.dtype,
            'min_value': np.min(data),
            'max_value': np.max(data),
            'mean_value': np.mean(data),
            'std_value': np.std(data),
            'has_nan': np.isnan(data).any(),
            'has_inf': np.isinf(data).any()
        }
        
        return info
    
    def load_ground_truth(self, dataset_name: str, gt_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load ground truth labels if available.
        
        Args:
            dataset_name: Name of the dataset
            gt_path: Optional path to ground truth file
            
        Returns:
            Ground truth array or None if not available
        """
        if gt_path and Path(gt_path).exists():
            try:
                gt_data = np.load(gt_path)
                self.logger.info(f"Loaded ground truth for {dataset_name}: {gt_data.shape}")
                return gt_data
            except Exception as e:
                self.logger.error(f"Failed to load ground truth for {dataset_name}: {str(e)}")
        
        return None
    
    def save_data(self, data: np.ndarray, save_path: str) -> None:
        """
        Save data to file.
        
        Args:
            data: Data to save
            save_path: Path to save the data
        """
        try:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            np.save(save_path, data)
            self.logger.info(f"Data saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {save_path}: {str(e)}")
            raise