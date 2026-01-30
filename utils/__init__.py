"""
Utilities for federated learning preprocessing and data handling.
"""

from .preprocessing import (
    HeartFailurePreprocessor,
    create_preprocessing_pipeline,
    load_and_preprocess_data,
)

__version__ = "0.1.0"
__all__ = [
    'HeartFailurePreprocessor',
    'create_preprocessing_pipeline',
    'load_and_preprocess_data',
]
