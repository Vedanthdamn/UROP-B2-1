"""
Preprocessing Pipeline for Federated Learning

This module provides a reusable, deterministic preprocessing pipeline for
federated training. The pipeline is designed to be used consistently across
all federated clients.

Key Features:
- Separates features and target label (DEATH_EVENT)
- Handles missing values safely
- Standardizes numerical features using StandardScaler
- Ensures deterministic and reproducible preprocessing
- Serializable and reusable during inference

Author: Federated Learning Medical AI Project
"""

import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Optional, Union


class HeartFailurePreprocessor:
    """
    Preprocessing pipeline for heart failure prediction dataset.
    
    This class provides a complete preprocessing pipeline that:
    1. Separates features from target label
    2. Handles missing values with median imputation
    3. Standardizes numerical features using z-score normalization
    4. Ensures consistency across train/test and federated clients
    
    The pipeline is designed to be serialized (via pickle) and reused
    across different federated clients and during inference.
    
    Attributes:
        target_column (str): Name of the target column (default: 'DEATH_EVENT')
        feature_means (np.ndarray): Mean values for each feature (computed during fit)
        feature_stds (np.ndarray): Standard deviations for each feature (computed during fit)
        feature_medians (np.ndarray): Median values for missing value imputation
        feature_columns (list): List of feature column names
        is_fitted (bool): Whether the preprocessor has been fitted
    """
    
    def __init__(self, target_column: str = 'DEATH_EVENT'):
        """
        Initialize the preprocessor.
        
        Args:
            target_column (str): Name of the target variable column.
                Default is 'DEATH_EVENT'.
        """
        self.target_column = target_column
        self.feature_means = None
        self.feature_stds = None
        self.feature_medians = None
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'HeartFailurePreprocessor':
        """
        Fit the preprocessor on training data.
        
        This computes and stores the statistics (means, standard deviations, medians)
        needed for preprocessing. These statistics are then used to transform any
        dataset consistently.
        
        Args:
            data (pd.DataFrame or np.ndarray): Training data containing both features
                and target column. If DataFrame, must include target_column.
                If ndarray, assumes last column is target.
        
        Returns:
            self: Returns self for method chaining.
        
        Raises:
            ValueError: If target column is not found or data is invalid.
        """
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            # Assume last column is target
            n_features = data.shape[1] - 1
            columns = [f'feature_{i}' for i in range(n_features)] + [self.target_column]
            data = pd.DataFrame(data, columns=columns)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame or numpy ndarray")
        
        # Check if target column exists
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data. "
                           f"Available columns: {data.columns.tolist()}")
        
        # Separate features from target
        X = data.drop(columns=[self.target_column])
        self.feature_columns = X.columns.tolist()
        
        # Convert to numpy for processing
        X_np = X.values.astype(np.float64)
        
        # Compute statistics for standardization
        # Note: We compute median for missing value imputation even though
        # the current dataset has no missing values
        self.feature_medians = np.nanmedian(X_np, axis=0)
        self.feature_means = np.nanmean(X_np, axis=0)
        self.feature_stds = np.nanstd(X_np, axis=0, ddof=0)
        
        # Prevent division by zero for constant features
        # Replace zero std with 1.0 to avoid scaling issues
        self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], 
                  return_target: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Transform data using the fitted preprocessing pipeline.
        
        This applies:
        1. Missing value imputation (median)
        2. Standardization (z-score normalization)
        
        Args:
            data (pd.DataFrame or np.ndarray): Data to transform.
            return_target (bool): Whether to return target variable.
                If True, returns (X, y). If False, returns only X.
                Default is True.
        
        Returns:
            If return_target=True:
                Tuple[np.ndarray, np.ndarray]: Transformed features (X) and target (y)
            If return_target=False:
                np.ndarray: Transformed features (X) only
        
        Raises:
            RuntimeError: If preprocessor has not been fitted.
            ValueError: If data format is invalid or target column is missing
                when return_target=True.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. "
                             "Call fit() first.")
        
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if return_target:
                # Assume last column is target
                columns = self.feature_columns + [self.target_column]
            else:
                columns = self.feature_columns
            data = pd.DataFrame(data, columns=columns)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame or numpy ndarray")
        
        # Separate features and target
        if return_target:
            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found. "
                               f"Available: {data.columns.tolist()}")
            X = data[self.feature_columns]
            y = data[self.target_column].values
        else:
            # For inference, we may not have target column
            X = data[self.feature_columns] if self.target_column in data.columns \
                else data
        
        # Convert to numpy
        X_np = X.values.astype(np.float64)
        
        # Step 1: Handle missing values (impute with median)
        # Create a mask for missing values
        missing_mask = np.isnan(X_np)
        if missing_mask.any():
            # Replace missing values with median
            for i in range(X_np.shape[1]):
                if missing_mask[:, i].any():
                    X_np[missing_mask[:, i], i] = self.feature_medians[i]
        
        # Step 2: Standardize features (z-score normalization)
        # Formula: X_scaled = (X - mean) / std
        X_standardized = (X_np - self.feature_means) / self.feature_stds
        
        if return_target:
            return X_standardized, y
        else:
            return X_standardized
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform data in one step.
        
        This is a convenience method that calls fit() followed by transform().
        
        Args:
            data (pd.DataFrame or np.ndarray): Training data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features (X) and target (y)
        """
        self.fit(data)
        return self.transform(data, return_target=True)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk.
        
        This enables the preprocessor to be reused across different federated
        clients and during inference.
        
        Args:
            filepath (str): Path where the preprocessor should be saved.
                Recommended extension: .pkl or .pickle
        
        Raises:
            RuntimeError: If preprocessor has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor. Call fit() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(filepath: str) -> 'HeartFailurePreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath (str): Path to the saved preprocessor file.
        
        Returns:
            HeartFailurePreprocessor: Loaded preprocessor instance.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        
        if not isinstance(preprocessor, HeartFailurePreprocessor):
            raise ValueError("Loaded object is not a HeartFailurePreprocessor instance")
        
        return preprocessor
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature column names.
        
        Returns:
            list: Feature column names, or None if not fitted.
        """
        return self.feature_columns if self.is_fitted else None
    
    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        if self.is_fitted:
            return (f"HeartFailurePreprocessor(target='{self.target_column}', "
                   f"n_features={len(self.feature_columns)}, fitted=True)")
        else:
            return f"HeartFailurePreprocessor(target='{self.target_column}', fitted=False)"


def create_preprocessing_pipeline(target_column: str = 'DEATH_EVENT') -> HeartFailurePreprocessor:
    """
    Factory function to create a new preprocessing pipeline.
    
    Args:
        target_column (str): Name of the target variable column.
            Default is 'DEATH_EVENT'.
    
    Returns:
        HeartFailurePreprocessor: A new preprocessor instance.
    
    Example:
        >>> preprocessor = create_preprocessing_pipeline()
        >>> X_train, y_train = preprocessor.fit_transform(train_data)
        >>> X_test, y_test = preprocessor.transform(test_data)
        >>> preprocessor.save('preprocessor.pkl')
    """
    return HeartFailurePreprocessor(target_column=target_column)


def load_and_preprocess_data(data_path: str, 
                            preprocessor: Optional[HeartFailurePreprocessor] = None,
                            fit: bool = True) -> Tuple[np.ndarray, np.ndarray, HeartFailurePreprocessor]:
    """
    Load data from CSV and preprocess it.
    
    This is a convenience function that combines data loading and preprocessing.
    
    Args:
        data_path (str): Path to the CSV file.
        preprocessor (HeartFailurePreprocessor, optional): Existing preprocessor.
            If None, creates a new one. Default is None.
        fit (bool): Whether to fit the preprocessor on this data.
            Set to True for training data, False for test data.
            Default is True.
    
    Returns:
        Tuple containing:
            - X (np.ndarray): Preprocessed features
            - y (np.ndarray): Target values
            - preprocessor (HeartFailurePreprocessor): The fitted preprocessor
    
    Example:
        >>> # For training data
        >>> X_train, y_train, preprocessor = load_and_preprocess_data('train.csv', fit=True)
        >>> 
        >>> # For test data (using same preprocessor)
        >>> X_test, y_test, _ = load_and_preprocess_data('test.csv', preprocessor=preprocessor, fit=False)
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Create preprocessor if not provided
    if preprocessor is None:
        preprocessor = create_preprocessing_pipeline()
    
    # Preprocess
    if fit:
        X, y = preprocessor.fit_transform(data)
    else:
        X, y = preprocessor.transform(data, return_target=True)
    
    return X, y, preprocessor
