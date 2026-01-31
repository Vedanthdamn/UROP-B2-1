"""
Preprocessing Pipeline for Federated Learning

This module provides a reusable, deterministic preprocessing pipeline for
federated training. The pipeline is designed to be used consistently across
all federated clients.

Key Features:
- Separates features and target label (DEATH_EVENT)
- Handles missing values safely
- Standardizes numerical features using z-score normalization
- Ensures deterministic and reproducible preprocessing
- Serializable and reusable during inference

Author: Federated Learning Medical AI Project
"""

import numpy as np
import pandas as pd
import pickle
import json
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
            # Validate that data contains expected feature columns
            missing_features = set(self.feature_columns) - set(data.columns)
            if missing_features:
                raise ValueError(f"Data is missing expected feature columns: {missing_features}. "
                               f"Expected: {self.feature_columns}")
            X = data[self.feature_columns]
        
        # Convert to numpy
        X_np = X.values.astype(np.float64)
        
        # Step 1: Handle missing values (impute with median)
        # Use NumPy broadcasting for efficient imputation
        if np.isnan(X_np).any():
            X_np = np.where(np.isnan(X_np), self.feature_medians, X_np)
        
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
    
    def save_safe(self, filepath: str) -> None:
        """
        Save the fitted preprocessor using JSON + NumPy arrays (safer alternative to pickle).
        
        This method provides safer serialization that is less susceptible to
        NumPy version incompatibilities. Recommended for inference-only code.
        
        Args:
            filepath (str): Path where the preprocessor should be saved.
                Recommended extension: .json
        
        Raises:
            RuntimeError: If preprocessor has not been fitted.
        
        Note:
            This saves preprocessing statistics as JSON + NPY arrays, avoiding
            pickle-related NumPy version incompatibilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor. Call fit() first.")
        
        # Get consistent file paths
        json_path, base_path = self._get_safe_paths(filepath)
        
        # Save metadata and statistics as JSON
        metadata = {
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        # Save JSON metadata
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save NumPy arrays separately
        np.save(f'{base_path}_means.npy', self.feature_means)
        np.save(f'{base_path}_stds.npy', self.feature_stds)
        np.save(f'{base_path}_medians.npy', self.feature_medians)
    
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
            ImportError: If incompatible NumPy version causes deserialization failure.
        
        Warning:
            Only load preprocessor files from trusted sources. Pickle can execute
            arbitrary code during deserialization, which could be a security risk
            if loading files from untrusted origins.
            
        Academic Note:
            If you encounter "ModuleNotFoundError: No module named 'numpy._core'",
            this indicates a NumPy version mismatch. NumPy 2.x changed internal
            module paths, breaking compatibility with artifacts serialized under
            NumPy 1.x. To fix this:
            1. Ensure NumPy is pinned to the same version used during training
            2. Regenerate preprocessing artifacts in the current environment
            3. Use save_safe()/load_safe() for safer serialization
        """
        try:
            with open(filepath, 'rb') as f:
                preprocessor = pickle.load(f)
            
            if not isinstance(preprocessor, HeartFailurePreprocessor):
                raise ValueError("Loaded object is not a HeartFailurePreprocessor instance")
            
            return preprocessor
        except (ImportError, ModuleNotFoundError) as e:
            # Handle NumPy version incompatibility
            if 'numpy._core' in str(e) or 'numpy.core' in str(e):
                raise ImportError(
                    "\n" + "=" * 80 + "\n"
                    "INCOMPATIBLE PREPROCESSING ARTIFACT DETECTED\n"
                    "=" * 80 + "\n\n"
                    "Error: The preprocessing artifact was created with a different NumPy version.\n\n"
                    "Root Cause:\n"
                    "  NumPy 2.x introduced breaking changes to internal module paths,\n"
                    "  making pickle artifacts serialized under NumPy 1.x incompatible.\n\n"
                    "Resolution:\n"
                    "  1. Ensure requirements.txt specifies the correct NumPy version (e.g., numpy==1.26.4)\n"
                    "  2. Reinstall dependencies: pip install -r requirements.txt\n"
                    "  3. Regenerate preprocessing artifacts in the pinned environment:\n"
                    "     - Delete old .pkl files\n"
                    "     - Re-run preprocessing: python demo_preprocessing.py\n"
                    "  4. Alternatively, use save_safe()/load_safe() for version-independent serialization\n\n"
                    "Academic Context:\n"
                    "  This is a known reproducibility issue in research systems.\n"
                    "  ML artifacts must be regenerated when dependencies change to ensure\n"
                    "  deterministic and verifiable results.\n\n"
                    f"Original Error: {str(e)}\n"
                    + "=" * 80
                ) from e
            else:
                # Re-raise other import errors
                raise
    
    @staticmethod
    def load_safe(filepath: str) -> 'HeartFailurePreprocessor':
        """
        Load a fitted preprocessor from JSON + NumPy arrays (safer alternative to pickle).
        
        This method provides safer deserialization that is immune to
        NumPy version incompatibilities.
        
        Args:
            filepath (str): Path to the saved preprocessor JSON file.
        
        Returns:
            HeartFailurePreprocessor: Loaded preprocessor instance.
        
        Raises:
            FileNotFoundError: If required files are not found.
        """
        # Get consistent file paths
        json_path, base_path = HeartFailurePreprocessor._get_safe_paths(filepath)
        
        # Load JSON metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Create preprocessor instance
        preprocessor = HeartFailurePreprocessor(target_column=metadata['target_column'])
        preprocessor.feature_columns = metadata['feature_columns']
        preprocessor.is_fitted = metadata['is_fitted']
        
        # Load NumPy arrays
        preprocessor.feature_means = np.load(f'{base_path}_means.npy')
        preprocessor.feature_stds = np.load(f'{base_path}_stds.npy')
        preprocessor.feature_medians = np.load(f'{base_path}_medians.npy')
        
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
    
    @staticmethod
    def _get_safe_paths(filepath: str) -> tuple:
        """
        Get consistent file paths for safe serialization.
        
        Args:
            filepath: Base filepath (with or without .json extension)
        
        Returns:
            Tuple of (json_path, base_path) for consistent file naming
        """
        json_path = filepath if filepath.endswith('.json') else filepath + '.json'
        base_path = json_path.replace('.json', '')
        return json_path, base_path


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
