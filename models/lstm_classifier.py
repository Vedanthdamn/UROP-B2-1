"""
LSTM-based Classifier for Federated Medical Classification (PRIMARY MODEL)

This module implements a shallow LSTM-based neural network architecture for
heart failure prediction. It is designed as the PRIMARY model for federated
learning with differential privacy.

Design Principles:
- Shallow architecture (minimal layers) for FL-friendliness
- Low parameter count to reduce communication costs
- Compatible with differential privacy training
- Suitable for tabular medical data with temporal patterns

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


class LSTMClassifier:
    """
    LSTM-based binary classifier for federated medical classification.
    
    This is the PRIMARY model for federated training. The architecture uses:
    - A single LSTM layer with dropout for regularization
    - A dense output layer with sigmoid activation
    - Shallow design to minimize parameters
    
    The model is optimized for:
    - Federated learning scenarios (low communication overhead)
    - Differential privacy training
    - Edge device deployment
    
    Attributes:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate for regularization
        model (tf.keras.Model): The compiled Keras model
    """
    
    def __init__(self, input_shape=(1, 12), lstm_units=32, dropout_rate=0.3):
        """
        Initialize the LSTM classifier.
        
        Args:
            input_shape (tuple): Input shape (sequence_length, n_features).
                Default is (1, 12) for heart failure dataset.
            lstm_units (int): Number of LSTM units. Default is 32 (shallow for FL).
            dropout_rate (float): Dropout rate for regularization. Default is 0.3.
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the LSTM classifier architecture.
        
        Architecture:
        1. Input layer: (sequence_length, n_features)
        2. LSTM layer: lstm_units with return_sequences=False
        3. Dropout layer: dropout_rate
        4. Dense output layer: 1 unit with sigmoid activation
        
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input')
        
        # LSTM layer (shallow, FL-friendly)
        # return_sequences=False: only return output at last timestep
        x = layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,
            name='lstm'
        )(inputs)
        
        # Dropout for regularization
        x = layers.Dropout(rate=self.dropout_rate, name='dropout')(x)
        
        # Output layer: binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTMClassifier')
        
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
        print("LSTM CLASSIFIER - PRIMARY MODEL FOR FEDERATED LEARNING")
        print("=" * 80)
        self.model.summary()
        print("=" * 80)
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Input shape: (batch_size, {self.input_shape[0]}, {self.input_shape[1]})")
        print(f"Output shape: (batch_size, 1)")
        print(f"LSTM units: {self.lstm_units}")
        print(f"Dropout rate: {self.dropout_rate}")
        print("=" * 80)
        print("\nDesign Notes:")
        print("  - Shallow architecture minimizes parameters for FL communication efficiency")
        print("  - Single LSTM layer captures temporal patterns in clinical data")
        print("  - Dropout provides regularization for better generalization")
        print("  - Compatible with differential privacy training")
        print("  - Suitable for edge device deployment")
        print("=" * 80 + "\n")


def create_lstm_classifier(input_shape=(1, 12), lstm_units=32, dropout_rate=0.3):
    """
    Factory function to create an LSTM classifier.
    
    Args:
        input_shape (tuple): Input shape (sequence_length, n_features).
            Default is (1, 12) for heart failure dataset.
        lstm_units (int): Number of LSTM units. Default is 32.
        dropout_rate (float): Dropout rate. Default is 0.3.
    
    Returns:
        tf.keras.Model: Compiled LSTM classifier.
    
    Example:
        >>> model = create_lstm_classifier(input_shape=(1, 12))
        >>> model.summary()
        >>> # Train with federated learning
        >>> # model.fit(X_train, y_train, ...)
    """
    classifier = LSTMClassifier(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    return classifier.get_model()


def get_model_info():
    """
    Get information about the LSTM classifier.
    
    Returns:
        dict: Model information including parameter count and design notes.
    """
    # Create a temporary model to get parameter count
    temp_model = create_lstm_classifier()
    param_count = temp_model.count_params()
    
    return {
        'name': 'LSTM Classifier',
        'type': 'PRIMARY',
        'architecture': 'LSTM-based',
        'input_shape': '(batch_size, 1, 12)',
        'output_shape': '(batch_size, 1)',
        'parameters': param_count,
        'lstm_units': 32,
        'dropout_rate': 0.3,
        'fl_friendly': True,
        'dp_compatible': True,
        'use_case': 'Primary model for federated training',
        'design_notes': [
            'Shallow architecture for FL communication efficiency',
            'Single LSTM layer for temporal pattern learning',
            'Dropout regularization for generalization',
            'Compatible with differential privacy',
            'Optimized for edge devices'
        ]
    }
