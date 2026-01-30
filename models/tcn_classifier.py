"""
Temporal Convolutional Network (TCN) Classifier for Federated Medical Classification

This module implements a lightweight Temporal Convolutional Network architecture
for heart failure prediction. It uses causal convolutions to process temporal
patterns in clinical data while maintaining a shallow, FL-friendly design.

Design Principles:
- Shallow TCN architecture with dilated causal convolutions
- Low parameter count for federated learning efficiency
- Compatible with differential privacy training
- Suitable for tabular medical data with temporal dependencies

Input Shape:
- (batch_size, sequence_length, n_features)
- For heart failure dataset: (batch_size, 1, 12)
  - sequence_length=1: treating tabular data as single timestep
  - n_features=12: preprocessed clinical features

Output:
- Binary classification (0 or 1) for DEATH_EVENT prediction

Author: Federated Learning Medical AI Project
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TCNClassifier:
    """
    Temporal Convolutional Network (TCN) binary classifier for federated medical classification.
    
    The architecture uses:
    - Causal 1D convolutions to respect temporal ordering
    - Dilated convolutions for expanded receptive field
    - Batch normalization for training stability
    - Shallow design to minimize parameters
    
    The model is optimized for:
    - Federated learning scenarios (low communication overhead)
    - Differential privacy training
    - Edge device deployment
    
    Attributes:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        filters (int): Number of convolutional filters
        kernel_size (int): Size of convolutional kernels
        dropout_rate (float): Dropout rate for regularization
        model (tf.keras.Model): The compiled Keras model
    """
    
    def __init__(self, input_shape=(1, 12), filters=32, kernel_size=2, dropout_rate=0.3):
        """
        Initialize the TCN classifier.
        
        Args:
            input_shape (tuple): Input shape (sequence_length, n_features).
                Default is (1, 12) for heart failure dataset.
            filters (int): Number of convolutional filters. Default is 32.
            kernel_size (int): Size of convolutional kernels. Default is 2.
            dropout_rate (float): Dropout rate for regularization. Default is 0.3.
        """
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the TCN classifier architecture.
        
        Architecture:
        1. Input layer: (sequence_length, n_features)
        2. Causal Conv1D layer: filters with dilation_rate=1
        3. Batch Normalization
        4. ReLU activation
        5. Dropout layer
        6. Global Average Pooling 1D
        7. Dense output layer: 1 unit with sigmoid activation
        
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input')
        
        # TCN block 1: Causal convolution with dilation
        # padding='causal' ensures future information doesn't leak
        x = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='causal',
            dilation_rate=1,
            name='tcn_conv1'
        )(inputs)
        
        # Batch normalization for training stability
        x = layers.BatchNormalization(name='batch_norm')(x)
        
        # Activation
        x = layers.Activation('relu', name='relu')(x)
        
        # Dropout for regularization
        x = layers.Dropout(rate=self.dropout_rate, name='dropout')(x)
        
        # Global pooling to reduce to single output per sample
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Output layer: binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='TCNClassifier')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def get_model(self):
        """
        Get the Keras model.
        
        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        return self.model
    
    def summary(self):
        """
        Print the model summary including architecture and parameter count.
        """
        print("\n" + "=" * 80)
        print("TEMPORAL CONVOLUTIONAL NETWORK (TCN) CLASSIFIER")
        print("=" * 80)
        self.model.summary()
        print("=" * 80)
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Input shape: (batch_size, {self.input_shape[0]}, {self.input_shape[1]})")
        print(f"Output shape: (batch_size, 1)")
        print(f"Filters: {self.filters}")
        print(f"Kernel size: {self.kernel_size}")
        print(f"Dropout rate: {self.dropout_rate}")
        print("=" * 80)
        print("\nDesign Notes:")
        print("  - Causal convolutions respect temporal ordering")
        print("  - Shallow architecture minimizes parameters for FL")
        print("  - Batch normalization stabilizes training")
        print("  - Global pooling reduces dimensionality efficiently")
        print("  - Compatible with differential privacy training")
        print("=" * 80 + "\n")


def create_tcn_classifier(input_shape=(1, 12), filters=32, kernel_size=2, dropout_rate=0.3):
    """
    Factory function to create a TCN classifier.
    
    Args:
        input_shape (tuple): Input shape (sequence_length, n_features).
            Default is (1, 12) for heart failure dataset.
        filters (int): Number of convolutional filters. Default is 32.
        kernel_size (int): Size of convolutional kernels. Default is 2.
        dropout_rate (float): Dropout rate. Default is 0.3.
    
    Returns:
        tf.keras.Model: Compiled TCN classifier.
    
    Example:
        >>> model = create_tcn_classifier(input_shape=(1, 12))
        >>> model.summary()
        >>> # Train with federated learning
        >>> # model.fit(X_train, y_train, ...)
    """
    classifier = TCNClassifier(
        input_shape=input_shape,
        filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    )
    return classifier.get_model()


def get_model_info():
    """
    Get information about the TCN classifier.
    
    Returns:
        dict: Model information including parameter count and design notes.
    """
    # Create a temporary model to get parameter count
    temp_model = create_tcn_classifier()
    param_count = temp_model.count_params()
    
    return {
        'name': 'TCN Classifier',
        'type': 'COMPARATIVE',
        'architecture': 'Temporal Convolutional Network',
        'input_shape': '(batch_size, 1, 12)',
        'output_shape': '(batch_size, 1)',
        'parameters': param_count,
        'filters': 32,
        'kernel_size': 2,
        'dropout_rate': 0.3,
        'fl_friendly': True,
        'dp_compatible': True,
        'use_case': 'Comparative model for benchmarking',
        'design_notes': [
            'Causal convolutions for temporal ordering',
            'Shallow architecture for FL efficiency',
            'Batch normalization for training stability',
            'Global pooling for dimensionality reduction',
            'Compatible with differential privacy'
        ]
    }
