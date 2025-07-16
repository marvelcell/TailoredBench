"""
Data processing utilities for model performance prediction.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from config.settings import PREPROCESSING_CONFIG, VALIDATION_CONFIG, get_dataset_config


class DataProcessor:
    """Handles data preprocessing and score calculations."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
    
    def get_gsm8k_true_scores(self, target_models_golddata: np.ndarray) -> List[float]:
        """
        Calculate true scores for gsm8k/winogrande/POPE datasets.
        
        Args:
            target_models_golddata: Target model data array
            
        Returns:
            List of true scores for each model
        """
        true_score = []
        item_num = target_models_golddata.shape[1]
        
        for data in target_models_golddata:
            greater_than_threshold = data > PREPROCESSING_CONFIG['threshold']
            # Calculate the number of elements that meet the condition
            count = np.sum(greater_than_threshold)
            true_score.append(count / item_num)
        
        return true_score
    
    def get_arc_hellaswag_true_scores(self, target_all_data: np.ndarray, gt_label: np.ndarray) -> List[float]:
        true_scores = []
        num_questions = target_all_data.shape[1]
        
        for model_idx in range(target_all_data.shape[0]):
            correct_count = 0
            for question_idx in range(num_questions):
                confidence = target_all_data[model_idx, question_idx]
                if isinstance(confidence, np.ndarray):
                    chosen_option = np.argmax(confidence)
                    if chosen_option == gt_label[question_idx]:
                        correct_count += 1
                else:
                    self.logger.warning(f"Invalid data at model {model_idx}, question {question_idx}")
            true_scores.append(correct_count / num_questions)
        
        return true_scores
    
    def get_true_scores(self, all_data: np.ndarray, gt: np.ndarray, 
                       metric: Any = accuracy_score) -> List[float]:
        """
        Calculate true scores using a specified metric.
        
        Args:
            all_data: Model prediction data
            gt: Ground truth labels
            metric: Metric function to use (default: accuracy_score)
            
        Returns:
            List of true scores for each model
        """
        gt = gt[:all_data.shape[1]]
        all_predicted_classes = np.argmax(all_data, axis=-1)
        
        all_true_scores = [
            metric(gt, all_predicted_classes[i]) for i in range(all_data.shape[0])
        ]
        
        return all_true_scores
    
    def get_variant_true_scores(self, all_data: np.ndarray, gt: np.ndarray, 
                               metric: Any = accuracy_score) -> List[float]:
        """
        Calculate true scores when inner elements have inconsistent quantities.
        
        Args:
            all_data: Model prediction data with variant inner dimensions
            gt: Ground truth labels
            metric: Metric function to use (default: accuracy_score)
            
        Returns:
            List of true scores for each model
        """
        all_predicted_classes = np.zeros((all_data.shape[0], all_data.shape[1]), dtype=int)
        
        # Iterate through all_data to find the index of maximum value for each innermost array
        for i in range(all_data.shape[0]):
            for j in range(all_data.shape[1]):
                # If all_data[i, j] is an array, find the index of its maximum value
                if isinstance(all_data[i, j], np.ndarray):
                    all_predicted_classes[i, j] = np.argmax(all_data[i, j])
                else:
                    # Handle possible exceptions, e.g., all_data[i, j] is not an array
                    self.logger.warning("Exception occurred in get_variant_true_scores")
                    all_predicted_classes[i, j] = -1  # Or other default value
        
        gt = gt[:all_data.shape[1]]
        all_true_scores = [
            metric(gt, all_predicted_classes[i]) for i in range(all_data.shape[0])
        ]
        
        return all_true_scores
    
    def compute_acc_data(self, all_data: np.ndarray, gt_label: np.ndarray) -> np.ndarray:
        """
        Extract confidence scores for ground truth labels.
        
        Args:
            all_data: Model prediction data with confidence scores
            gt_label: Ground truth labels
                    
        Returns:
            Matrix containing confidence scores for correct answers (models x questions)
        """
        # Get the number of models and questions
        num_models, num_questions = all_data.shape[:2]
                
        # Initialize a 2D golddata matrix
        golddata = np.zeros((num_models, num_questions))
                
        # Iterate through each model and each question
        for model_idx in range(num_models):
            for question_idx in range(num_questions):
                # Get model confidence for each option
                confidence = all_data[model_idx, question_idx]
                                
                # Extract the confidence score for the correct answer
                golddata[model_idx, question_idx] = confidence[gt_label[question_idx]]
                
        return golddata
    
    def standardize_data(self, data: np.ndarray, fit_transform: bool = True) -> np.ndarray:
        """
        Standardize data using StandardScaler.
        
        Args:
            data: Input data to standardize
            fit_transform: Whether to fit and transform or just transform
            
        Returns:
            Standardized data
        """
        if not PREPROCESSING_CONFIG['standardization']:
            return data
        
        if fit_transform:
            return self.scaler.fit_transform(data)
        else:
            return self.scaler.transform(data)
    
    def apply_score_bounds(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply score bounds to ensure values are within valid range.
        
        Args:
            scores: Input scores
            
        Returns:
            Bounded scores
        """
        min_val = PREPROCESSING_CONFIG['score_bounds']['min']
        max_val = PREPROCESSING_CONFIG['score_bounds']['max']
        
        scores = np.clip(scores, min_val, max_val)
        return scores
    
    def handle_nan_values(self, data: np.ndarray, strategy: str = 'zero') -> np.ndarray:
        """
        Handle NaN values in data.
        
        Args:
            data: Input data
            strategy: Strategy for handling NaN values ('zero', 'mean', 'remove')
            
        Returns:
            Data with NaN values handled
        """
        if not VALIDATION_CONFIG['handle_nan_values']:
            return data
        
        if not np.isnan(data).any():
            return data
        
        if strategy == 'zero':
            data = np.nan_to_num(data, nan=0.0)
        elif strategy == 'mean':
            data = np.nan_to_num(data, nan=np.nanmean(data))
        elif strategy == 'remove':
            # This would require more complex logic depending on data structure
            self.logger.warning("NaN removal strategy not implemented for this data structure")
        
        return data
    
    def preprocess_model_data(self, data: np.ndarray, transpose: bool = False, 
                            standardize: bool = True) -> np.ndarray:
        """
        Preprocess model data with common operations.
        
        Args:
            data: Input data
            transpose: Whether to transpose the data
            standardize: Whether to standardize the data
            
        Returns:
            Preprocessed data
        """
        processed_data = data.copy()
        
        # Handle NaN values
        processed_data = self.handle_nan_values(processed_data)
        
        # Transpose if needed
        if transpose:
            processed_data = processed_data.T
        
        # Standardize if needed
        if standardize:
            processed_data = self.standardize_data(processed_data)
        
        return processed_data
    
    def calculate_score_adjustment(self, observe_data: np.ndarray, 
                                 means_data: np.ndarray) -> np.ndarray:
        """
        Calculate score adjustment factors for cluster correction.
        
        Args:
            observe_data: Observed centroid data
            means_data: Mean centroid data
            
        Returns:
            Adjustment factors
        """
        # Add offset to avoid division by zero
        observe_data = observe_data.astype(np.float64) + 0.5
        means_data = means_data.astype(np.float64) + 0.5
        
        # Calculate correction scale
        correction_scale = observe_data / means_data
        
        return correction_scale
    
    def apply_cluster_correction(self, data: np.ndarray, clustered_indices: List[List[int]], 
                               correction_scales: np.ndarray) -> np.ndarray:
        """
        Apply cluster-based correction to data.
        
        Args:
            data: Input data to correct
            clustered_indices: List of indices for each cluster
            correction_scales: Correction factors for each cluster
            
        Returns:
            Corrected data
        """
        corrected_data = data.copy()
        
        # Apply correction for each cluster
        for i, cluster_indices in enumerate(clustered_indices):
            if cluster_indices:  # Check if cluster is not empty
                corrected_data[cluster_indices] *= correction_scales[i]
        
        return corrected_data
    
    def finalize_scores(self, scores: np.ndarray, offset: float = 0.5) -> np.ndarray:
        """
        Finalize scores by removing offset and applying bounds.
        
        Args:
            scores: Input scores
            offset: Offset to remove
            
        Returns:
            Finalized scores
        """
        # Remove offset
        scores = scores - offset
        
        # Apply bounds
        scores = self.apply_score_bounds(scores)
        
        return scores
    

    def get_dataset_specific_scores(self, dataset_name: str, 
                                  target_models_golddata: np.ndarray, 
                                  target_all_data: Optional[np.ndarray] = None, 
                                  gt_label: Optional[np.ndarray] = None) -> List[float]:
        """
        Get true scores based on dataset type.
        
        Args:
            dataset_name: Name of the dataset
            target_models_golddata: Target model data
            
        Returns:
            List of true scores
        """
        config = get_dataset_config(dataset_name)
        if config['requires_gt']:
            if target_all_data is None or gt_label is None:
                raise ValueError(f"Dataset {dataset_name} requires target_all_data and gt_label")
            return self.get_arc_hellaswag_true_scores(target_all_data, gt_label)
        elif dataset_name in ['gsm8k', 'winogrande', 'POPE']:
            return self.get_gsm8k_true_scores(target_models_golddata)
        else:
            self.logger.warning(f"Unknown dataset: {dataset_name}, using gsm8k method")
            return self.get_gsm8k_true_scores(target_models_golddata)
    
    def validate_data_consistency(self, source_data: np.ndarray, 
                                target_data: np.ndarray) -> bool:
        """
        Validate consistency between source and target data.
        
        Args:
            source_data: Source model data
            target_data: Target model data
            
        Returns:
            True if data is consistent, False otherwise
        """
        if source_data.shape[1] != target_data.shape[1]:
            self.logger.error("Source and target data have different number of samples")
            return False
        
        if source_data.ndim != target_data.ndim:
            self.logger.error("Source and target data have different dimensions")
            return False
        
        return True