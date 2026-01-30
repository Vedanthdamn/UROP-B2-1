# Model Architectures for Federated Medical Classification

This directory contains TensorFlow/Keras model architectures designed for federated learning on tabular medical data (heart failure prediction).

## Overview

All models are designed with the following principles:
- **Shallow architectures**: Minimal layers to reduce parameters
- **FL-friendly**: Low communication overhead for federated learning
- **DP-compatible**: Compatible with differential privacy training
- **Edge-optimized**: Suitable for deployment on edge devices

## Available Models

### 1. LSTM Classifier (PRIMARY MODEL) ⭐

**File**: `lstm_classifier.py`

**Architecture**:
- Single LSTM layer (32 units)
- Dropout layer (0.3 rate)
- Dense output layer (sigmoid activation)

**Input Shape**: `(batch_size, 1, 12)`
- sequence_length = 1 (treating tabular data as single timestep)
- n_features = 12 (preprocessed clinical features)

**Output Shape**: `(batch_size, 1)` - Binary classification for DEATH_EVENT

**Use Case**: **This is the PRIMARY model for federated training**. Use this model for the main federated learning experiments.

**Design Rationale**:
- LSTM naturally handles temporal patterns in clinical data
- Shallow single-layer design minimizes parameters for FL efficiency
- Dropout provides regularization without adding parameters
- Well-suited for differential privacy training

**Example Usage**:
```python
from models import create_lstm_classifier

# Create model
model = create_lstm_classifier(input_shape=(1, 12))
model.summary()

# Prepare data (reshape to 3D if needed)
import numpy as np
X_train_reshaped = X_train.reshape(-1, 1, 12)  # (n_samples, 1, 12)

# Train (in federated setting)
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32)
```

### 2. Temporal Convolutional Network (TCN)

**File**: `tcn_classifier.py`

**Architecture**:
- Causal Conv1D layer (32 filters, kernel_size=2)
- Batch Normalization
- ReLU activation
- Dropout layer (0.3 rate)
- Global Average Pooling
- Dense output layer (sigmoid activation)

**Input Shape**: `(batch_size, 1, 12)`

**Output Shape**: `(batch_size, 1)`

**Use Case**: Comparative model for benchmarking against LSTM.

**Design Rationale**:
- Causal convolutions respect temporal ordering
- Batch normalization stabilizes training
- Global pooling efficiently reduces dimensionality
- Alternative to LSTM for sequence modeling

**Example Usage**:
```python
from models import create_tcn_classifier

model = create_tcn_classifier(input_shape=(1, 12))
model.summary()
```

### 3. Lightweight Transformer

**File**: `transformer_classifier.py`

**Architecture**:
- Multi-Head Attention (1 head, lightweight)
- Dropout layer
- Add & Norm (residual + layer normalization)
- Feed-Forward Network (32 units)
- Dropout layer
- Add & Norm
- Global Average Pooling
- Dense output layer (sigmoid activation)

**Input Shape**: `(batch_size, 1, 12)`

**Output Shape**: `(batch_size, 1)`

**Use Case**: Comparative model for benchmarking against LSTM and TCN.

**Design Rationale**:
- Single-head attention keeps parameters minimal
- Residual connections improve gradient flow
- Layer normalization provides training stability
- Modern architecture for comparative analysis

**Example Usage**:
```python
from models import create_transformer_classifier

model = create_transformer_classifier(input_shape=(1, 12))
model.summary()
```

## Model Comparison

| Model | Type | Parameters | Key Feature | FL Efficiency |
|-------|------|------------|-------------|---------------|
| **LSTM** | **PRIMARY** | ~5K-6K | Temporal patterns | **High** |
| TCN | Comparative | ~1K | Causal convolutions | Very High |
| Transformer | Comparative | ~1K-2K | Self-attention | High |

*Note: Exact parameter counts depend on input dimensions and hyperparameters*

## Usage Patterns

### Pattern 1: Using the Primary Model (Recommended)

```python
from models import get_primary_model

# Get LSTM classifier (primary model)
model = get_primary_model(input_shape=(1, 12))
model.summary()
```

### Pattern 2: Using Model Registry

```python
from models import get_model

# Get specific model by name
lstm_model = get_model('lstm', input_shape=(1, 12))
tcn_model = get_model('tcn', input_shape=(1, 12))
transformer_model = get_model('transformer', input_shape=(1, 12))
```

### Pattern 3: Direct Import

```python
from models import LSTMClassifier, TCNClassifier, TransformerClassifier

# Create model objects
lstm = LSTMClassifier(input_shape=(1, 12), lstm_units=32)
lstm.summary()  # Print detailed summary

# Get Keras model
model = lstm.get_model()
```

## Integration with Preprocessing Pipeline

The models are designed to work seamlessly with the preprocessing pipeline:

```python
from utils.preprocessing import load_and_preprocess_data
from models import get_primary_model
import numpy as np

# Load and preprocess data
X, y, preprocessor = load_and_preprocess_data('data/heart_failure.csv', fit=True)

# Reshape for model input (add sequence dimension)
X_reshaped = X.reshape(-1, 1, 12)  # (n_samples, 1, 12)

# Create and train model
model = get_primary_model(input_shape=(1, 12))
model.fit(X_reshaped, y, epochs=10, batch_size=32, validation_split=0.2)
```

## Federated Learning Considerations

### Communication Efficiency

All models use shallow architectures to minimize:
- **Model size**: Fewer parameters = smaller model updates
- **Communication rounds**: Faster convergence with efficient models
- **Bandwidth usage**: Less data transmitted between clients and server

### Differential Privacy Compatibility

All models are compatible with differential privacy training:
- Gradient clipping can be applied during training
- Noise can be added to gradients without architecture changes
- Shallow architectures reduce sensitivity to noise

### Edge Device Deployment

Models are designed for edge devices:
- Low memory footprint
- Fast inference time
- Compatible with TensorFlow Lite for mobile deployment

## Model Selection Guide

**For Federated Training (Production)**:
- ✅ **Use LSTM Classifier (PRIMARY)**
- Reason: Best balance of performance, efficiency, and FL-friendliness

**For Benchmarking and Research**:
- Use all three models to compare:
  - LSTM: Baseline primary model
  - TCN: Convolutional alternative
  - Transformer: Modern attention-based approach

**For Specific Requirements**:
- Need minimal parameters → TCN
- Need self-attention → Transformer
- Need temporal modeling → LSTM (PRIMARY)

## Testing

Run the test suite to validate all models:

```bash
python test_models.py
```

Run the demo script to see usage examples:

```bash
python demo_models.py
```

## Parameter Counts

Run this script to see exact parameter counts:

```python
from models import get_model

for model_name in ['lstm', 'tcn', 'transformer']:
    model = get_model(model_name, input_shape=(1, 12))
    print(f"{model_name.upper()}: {model.count_params():,} parameters")
```

## Customization

All models support hyperparameter customization:

```python
# LSTM with more units
model = create_lstm_classifier(input_shape=(1, 12), lstm_units=64, dropout_rate=0.5)

# TCN with more filters
model = create_tcn_classifier(input_shape=(1, 12), filters=64, kernel_size=3)

# Transformer with more heads
model = create_transformer_classifier(input_shape=(1, 12), num_heads=2, ff_dim=64)
```

**Note**: Increasing parameters may reduce FL efficiency. Keep architectures shallow for production use.

## Important Notes

1. **No Training Required**: This directory only contains model definitions. Training is done separately in federated learning experiments.

2. **Input Shape**: All models expect 3D input `(batch_size, sequence_length, n_features)`. For tabular data, use `sequence_length=1`.

3. **Data Preprocessing**: Always use the `HeartFailurePreprocessor` from `utils.preprocessing` before feeding data to models.

4. **Primary Model**: The LSTM Classifier is designated as the PRIMARY model for federated training.

## File Structure

```
models/
├── __init__.py                 # Package initialization and model registry
├── lstm_classifier.py          # LSTM Classifier (PRIMARY)
├── tcn_classifier.py           # TCN Classifier
├── transformer_classifier.py   # Transformer Classifier
└── README.md                   # This file
```

## Version

Current version: 1.0.0

Last updated: 2026-01-30

## Authors

Federated Learning Medical AI Project
