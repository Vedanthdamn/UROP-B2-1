"""
Lightweight Transformer Classifier for Federated Medical Classification

This module implements a simplified Transformer-based architecture for heart
failure prediction. It uses self-attention mechanisms while maintaining a
shallow, FL-friendly design for comparative analysis.

Design Principles:
- Lightweight single-head attention mechanism
- Minimal layers to reduce parameter count
- Compatible with differential privacy training
- Suitable for tabular medical data

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


class TransformerClassifier:
    """
    Lightweight Transformer-based binary classifier for federated medical classification.
    
    This is a COMPARATIVE model for benchmarking against LSTM. The architecture uses:
    - Single-head attention (lightweight)
    - Layer normalization for training stability
    - Feed-forward network
    - Shallow design to minimize parameters
    
    The model is optimized for:
    - Federated learning scenarios (low communication overhead)
    - Differential privacy training
    - Comparative analysis with LSTM and TCN
    
    Attributes:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        num_heads (int): Number of attention heads
        ff_dim (int): Dimension of feed-forward network
        dropout_rate (float): Dropout rate for regularization
        model (tf.keras.Model): The compiled Keras model
    """
    
    def __init__(self, input_shape=(1, 12), num_heads=1, ff_dim=32, dropout_rate=0.3):
        """
        Initialize the Transformer classifier.
        
        Args:
            input_shape (tuple): Input shape (sequence_length, n_features).
                Default is (1, 12) for heart failure dataset.
            num_heads (int): Number of attention heads. Default is 1 (lightweight).
            ff_dim (int): Dimension of feed-forward network. Default is 32.
            dropout_rate (float): Dropout rate for regularization. Default is 0.3.
        """
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the lightweight Transformer classifier architecture.
        
        Architecture:
        1. Input layer: (sequence_length, n_features)
        2. Multi-Head Attention layer (single head for simplicity)
        3. Dropout
        4. Add & Norm (residual connection + layer normalization)
        5. Feed-Forward Network
        6. Dropout
        7. Add & Norm
        8. Global Average Pooling
        9. Dense output layer: 1 unit with sigmoid activation
        
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input')
        
        # Multi-head attention (lightweight: 1 head)
        # key_dim is set to input dimension divided by num_heads
        key_dim = self.input_shape[1] // self.num_heads
        if key_dim < 1:
            key_dim = self.input_shape[1]
        
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name='multihead_attention'
        )(inputs, inputs)
        
        # Dropout
        attention_output = layers.Dropout(self.dropout_rate, name='attention_dropout')(attention_output)
        
        # Add & Norm (residual connection + layer normalization)
        x = layers.Add(name='add_1')([inputs, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')(x)
        
        # Feed-forward network
        ff_output = layers.Dense(self.ff_dim, activation='relu', name='ff_dense_1')(x)
        ff_output = layers.Dropout(self.dropout_rate, name='ff_dropout')(ff_output)
        ff_output = layers.Dense(self.input_shape[1], name='ff_dense_2')(ff_output)
        
        # Add & Norm (residual connection + layer normalization)
        x = layers.Add(name='add_2')([x, ff_output])
        x = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_2')(x)
        
        # Global pooling to reduce to single output per sample
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Output layer: binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='TransformerClassifier')
        
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
        print("LIGHTWEIGHT TRANSFORMER CLASSIFIER (COMPARATIVE MODEL)")
        print("=" * 80)
        self.model.summary()
        print("=" * 80)
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Input shape: (batch_size, {self.input_shape[0]}, {self.input_shape[1]})")
        print(f"Output shape: (batch_size, 1)")
        print(f"Number of attention heads: {self.num_heads}")
        print(f"Feed-forward dimension: {self.ff_dim}")
        print(f"Dropout rate: {self.dropout_rate}")
        print("=" * 80)
        print("\nDesign Notes:")
        print("  - Single-head attention for lightweight design")
        print("  - Residual connections for better gradient flow")
        print("  - Layer normalization for training stability")
        print("  - Shallow architecture minimizes parameters for FL")
        print("  - Compatible with differential privacy training")
        print("  - Used for comparative benchmarking against LSTM")
        print("=" * 80 + "\n")


def create_transformer_classifier(input_shape=(1, 12), num_heads=1, ff_dim=32, dropout_rate=0.3):
    """
    Factory function to create a lightweight Transformer classifier.
    
    Args:
        input_shape (tuple): Input shape (sequence_length, n_features).
            Default is (1, 12) for heart failure dataset.
        num_heads (int): Number of attention heads. Default is 1.
        ff_dim (int): Dimension of feed-forward network. Default is 32.
        dropout_rate (float): Dropout rate. Default is 0.3.
    
    Returns:
        tf.keras.Model: Compiled Transformer classifier.
    
    Example:
        >>> model = create_transformer_classifier(input_shape=(1, 12))
        >>> model.summary()
        >>> # Train with federated learning
        >>> # model.fit(X_train, y_train, ...)
    """
    classifier = TransformerClassifier(
        input_shape=input_shape,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate
    )
    return classifier.get_model()


def get_model_info():
    """
    Get information about the Transformer classifier.
    
    Returns:
        dict: Model information including parameter count and design notes.
    """
    # Create a temporary model to get parameter count
    temp_model = create_transformer_classifier()
    param_count = temp_model.count_params()
    
    return {
        'name': 'Transformer Classifier',
        'type': 'COMPARATIVE',
        'architecture': 'Lightweight Transformer',
        'input_shape': '(batch_size, 1, 12)',
        'output_shape': '(batch_size, 1)',
        'parameters': param_count,
        'num_heads': 1,
        'ff_dim': 32,
        'dropout_rate': 0.3,
        'fl_friendly': True,
        'dp_compatible': True,
        'use_case': 'Comparative model for benchmarking',
        'design_notes': [
            'Single-head attention for lightweight design',
            'Residual connections for gradient flow',
            'Layer normalization for stability',
            'Shallow architecture for FL efficiency',
            'Compatible with differential privacy',
            'Benchmarking against LSTM primary model'
        ]
    }
