# Preprocessing Pipeline Guide

## Overview

This guide explains the preprocessing pipeline implemented for federated learning in the heart failure prediction project.

## Architecture

### HeartFailurePreprocessor Class

The core component is the `HeartFailurePreprocessor` class located in `utils/preprocessing.py`. It provides:

1. **Feature/Target Separation**: Automatically separates the `DEATH_EVENT` target from features
2. **Missing Value Handling**: Uses median imputation (optimized with NumPy broadcasting)
3. **Standardization**: Z-score normalization (mean ≈ 0, std ≈ 1)
4. **Serialization**: Save/load via pickle for distributed deployment
5. **Determinism**: Ensures reproducible results across all federated clients

## Key Methods

### fit(data)
Computes preprocessing statistics from training data:
- Feature means
- Feature standard deviations
- Feature medians (for imputation)

### transform(data, return_target=True)
Applies preprocessing using computed statistics:
- Imputes missing values
- Standardizes features
- Returns (X, y) or X only

### fit_transform(data)
Convenience method: fit + transform in one call

### save(filepath) / load(filepath)
Serialize/deserialize the preprocessor for federated deployment

## Usage Patterns

### Pattern 1: Simple Usage

```python
from utils import create_preprocessing_pipeline

preprocessor = create_preprocessing_pipeline()
X_train, y_train = preprocessor.fit_transform(train_data)
X_test, y_test = preprocessor.transform(test_data)
```

### Pattern 2: Federated Learning

```python
from utils import HeartFailurePreprocessor, create_preprocessing_pipeline

# Central server: Fit on representative data
preprocessor = create_preprocessing_pipeline()
preprocessor.fit(global_training_data)
preprocessor.save('shared_preprocessor.pkl')

# Federated client: Load and use
preprocessor = HeartFailurePreprocessor.load('shared_preprocessor.pkl')
X_local, y_local = preprocessor.transform(local_data)
```

### Pattern 3: Inference (Production)

```python
from utils import HeartFailurePreprocessor

# Load trained preprocessor
preprocessor = HeartFailurePreprocessor.load('preprocessor.pkl')

# Transform new data (without target)
X_new = preprocessor.transform(new_data, return_target=False)
```

## Important Considerations

### 1. Consistency Across Clients

**CRITICAL**: All federated clients MUST use the SAME preprocessor instance (loaded from the same file). This ensures:
- Identical feature scaling
- Consistent missing value imputation
- Compatible model inputs

### 2. Data Format Requirements

The preprocessor accepts:
- **Pandas DataFrame**: Preferred, with column names
- **NumPy array**: Last column assumed to be target

For inference without target, use `return_target=False` and provide only feature columns.

### 3. Missing Value Strategy

The pipeline uses **median imputation** even though the current dataset has no missing values. This makes the pipeline robust to:
- Future data with missing values
- Different federated client datasets
- Real-world deployment scenarios

### 4. Constant Features

If a feature has zero variance (constant value), the preprocessor:
- Replaces std=0 with std=1 to avoid division by zero
- Standardizes the constant feature to 0

### 5. Security

**WARNING**: Only load preprocessor files from trusted sources. Pickle deserialization can execute arbitrary code if the file is malicious.

## Testing

### Run Complete Test Suite

```bash
python test_preprocessing.py
```

Tests cover:
1. Feature/target separation
2. Missing value handling
3. Standardization consistency
4. Serialization/deserialization
5. Determinism and reproducibility
6. Inference mode
7. Error handling
8. Helper functions
9. NumPy array input
10. Constant features
11. Column validation

### Run Demo

```bash
python demo_preprocessing.py
```

Shows practical usage examples including federated learning workflow.

## Technical Details

### Standardization Formula

```
X_scaled = (X - mean) / std
```

where:
- `mean` and `std` are computed from training data during `fit()`
- Applied consistently to all datasets during `transform()`

### Missing Value Imputation

```python
# Efficient NumPy broadcasting
X_imputed = np.where(np.isnan(X), medians, X)
```

### Serialization Format

- Uses Python's `pickle` module with highest protocol
- Stores entire preprocessor state (means, stds, medians, feature names)
- File size: ~few KB for 12 features

## Troubleshooting

### Issue: "Preprocessor must be fitted before transform"

**Solution**: Call `fit()` before `transform()`, or use `fit_transform()`

### Issue: "Target column 'DEATH_EVENT' not found"

**Solution**: 
- For training data: Include DEATH_EVENT column
- For inference: Use `return_target=False`

### Issue: "Data is missing expected feature columns"

**Solution**: Ensure inference data contains all feature columns that were present during training

### Issue: Different results across clients

**Solution**: Ensure all clients load the SAME preprocessor file (same version, not re-fitted)

## File Structure

```
utils/
├── __init__.py              # Package exports
└── preprocessing.py         # HeartFailurePreprocessor implementation

test_preprocessing.py        # Test suite (11 tests)
demo_preprocessing.py        # Usage examples
PREPROCESSING_GUIDE.md       # This file
```

## API Reference

See docstrings in `utils/preprocessing.py` for detailed API documentation.

Quick reference:
- `create_preprocessing_pipeline(target_column='DEATH_EVENT')` → HeartFailurePreprocessor
- `load_and_preprocess_data(path, preprocessor=None, fit=True)` → (X, y, preprocessor)

## Version

Current version: 0.1.0

Last updated: 2026-01-30
