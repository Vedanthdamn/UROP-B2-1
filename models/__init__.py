"""
Model architectures for federated medical classification on tabular data.

This module provides shallow, FL-friendly neural network architectures designed for:
- Heart failure prediction from tabular clinical data
- Federated learning with differential privacy
- Efficient training on edge devices

Available Models:
1. LSTMClassifier (PRIMARY): LSTM-based classifier for temporal patterns
2. TCNClassifier: Temporal Convolutional Network for sequence modeling
3. TransformerClassifier: Lightweight Transformer-based classifier (comparative)

All models:
- Accept input shape (batch_size, sequence_length, n_features) where n_features=12
- Use shallow architectures to minimize parameters (FL-friendly)
- Are compatible with differential privacy training
- Output binary classification predictions for DEATH_EVENT

Author: Federated Learning Medical AI Project
"""

from .lstm_classifier import LSTMClassifier, create_lstm_classifier
from .tcn_classifier import TCNClassifier, create_tcn_classifier
from .transformer_classifier import TransformerClassifier, create_transformer_classifier

__all__ = [
    'LSTMClassifier',
    'create_lstm_classifier',
    'TCNClassifier',
    'create_tcn_classifier',
    'TransformerClassifier',
    'create_transformer_classifier',
    'get_model',
    'get_primary_model',
    'MODEL_REGISTRY',
    'PRIMARY_MODEL',
]

# Model registry
MODEL_REGISTRY = {
    'lstm': create_lstm_classifier,
    'tcn': create_tcn_classifier,
    'transformer': create_transformer_classifier,
}

# Primary model for federated training
PRIMARY_MODEL = 'lstm'


def get_model(model_name='lstm', **kwargs):
    """
    Factory function to create a model by name.
    
    Args:
        model_name (str): Name of the model ('lstm', 'tcn', 'transformer').
            Default is 'lstm' (primary model).
        **kwargs: Additional arguments passed to the model constructor.
    
    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    
    Example:
        >>> model = get_model('lstm', input_shape=(1, 12))
        >>> model.summary()
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](**kwargs)


def get_primary_model(**kwargs):
    """
    Get the primary model for federated training.
    
    Returns:
        tf.keras.Model: The LSTM classifier (primary model).
    
    Example:
        >>> model = get_primary_model(input_shape=(1, 12))
        >>> model.summary()
    """
    return get_model(PRIMARY_MODEL, **kwargs)
