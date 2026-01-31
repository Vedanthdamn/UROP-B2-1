"""
Inference Pipeline for Federated Learning Medical AI

This module provides inference capabilities for the trained federated model.

Key Features:
- Load trained global federated model
- Load shared preprocessing pipeline
- Accept new patient input data (CSV)
- Apply preprocessing and run inference
- Output predicted disease label and confidence score

Constraints:
- NO training occurs during inference
- NO patient data is stored
- Uses trained federated model artifacts

Author: Federated Learning Medical AI Project
"""

from .inference_pipeline import InferencePipeline, predict_from_csv

__all__ = [
    'InferencePipeline',
    'predict_from_csv'
]
