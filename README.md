# UROP-B2-1

## Federated Learning Medical AI Project

This repository contains a federated learning project for medical AI, specifically for heart failure prediction.

## Dataset Validation

Before starting any federated learning experiments, you must validate the repository integrity and dataset availability.

### Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Validation Script

To validate the repository and dataset:

```bash
python validate_dataset.py
```

The validation script performs the following checks:

**Step 1: Repository Integrity Validation**
- Confirms that the `data/` folder exists
- Confirms that the `data/heart_failure.csv` file exists

**Step 2: Dataset Validation**
- Loads the dataset using pandas
- Displays the first 5 rows of the dataset
- Displays the dataset shape (rows and columns)

If all validations pass, the script exits with code 0. If any validation fails, the script reports the error and exits with code 1.

## Dataset Information

The `data/heart_failure.csv` dataset contains medical records for heart failure patients with the following features:

- **age**: Age of the patient
- **anaemia**: Decrease of red blood cells or hemoglobin (boolean)
- **creatinine_phosphokinase**: Level of the CPK enzyme in the blood (mcg/L)
- **diabetes**: If the patient has diabetes (boolean)
- **ejection_fraction**: Percentage of blood leaving the heart at each contraction
- **high_blood_pressure**: If the patient has hypertension (boolean)
- **platelets**: Platelets in the blood (kiloplatelets/mL)
- **serum_creatinine**: Level of serum creatinine in the blood (mg/dL)
- **serum_sodium**: Level of serum sodium in the blood (mEq/L)
- **sex**: Woman or man (binary)
- **smoking**: If the patient smokes or not (boolean)
- **time**: Follow-up period (days)
- **DEATH_EVENT**: If the patient deceased during the follow-up period (target variable)

## Dataset Sampling

The repository includes a deterministic dataset sampling module for creating reproducible training subsets for federated learning experiments.

### Features

- **Deterministic Sampling**: Uses fixed random seed for reproducibility
- **Stratified Sampling**: Preserves original class distribution
- **Comprehensive Logging**: Logs random seed and class distributions
- **Automatic Reporting**: Generates detailed markdown reports
- **Edge Case Handling**: Handles cases where sample size exceeds dataset size

### Usage

#### Basic Usage with Convenience Function

```python
from utils.data_sampling import sample_heart_failure_data

# Sample data and generate report
sampled_data = sample_heart_failure_data(
    'data/heart_failure.csv',
    n_samples=300,
    random_seed=42,
    output_report_path='reports/sampling_summary.md'
)
```

#### Advanced Usage with DatasetSampler Class

```python
from utils.data_sampling import DatasetSampler
import pandas as pd

# Load data
data = pd.read_csv('data/heart_failure.csv')

# Create sampler with custom seed
sampler = DatasetSampler(random_seed=123)

# Perform stratified sampling
sampled_data = sampler.sample(data, n_samples=150, stratify=True)

# Get sampling summary
summary = sampler.get_sampling_summary()
print(f"Distribution preserved: {summary['distribution_preserved']}")

# Generate report
sampler.generate_report('reports/custom_sampling.md')
```

### Testing and Demo

Run the test suite to validate the sampling functionality:

```bash
python test_sampling.py
```

Run the demo script to see usage examples:

```bash
python demo_sampling.py
```

## Preprocessing Pipeline

The repository includes a reusable preprocessing pipeline for federated learning, designed to ensure consistency across all federated clients.

### Features

- **Feature/Target Separation**: Automatically separates features from the target label (DEATH_EVENT)
- **Missing Value Handling**: Safely handles missing values using median imputation
- **Standardization**: Applies z-score normalization for consistent feature scaling
- **Serialization**: Pipeline can be saved and loaded for reuse across clients
- **Deterministic**: Ensures reproducible results across different runs
- **Inference Mode**: Supports preprocessing without target labels for inference

### Usage

#### Basic Usage

```python
from utils.preprocessing import create_preprocessing_pipeline

# Create and fit preprocessor
preprocessor = create_preprocessing_pipeline()
X_train, y_train = preprocessor.fit_transform(train_data)

# Transform test data
X_test, y_test = preprocessor.transform(test_data)
```

#### Convenience Function

```python
from utils.preprocessing import load_and_preprocess_data

# Load and preprocess in one step
X, y, preprocessor = load_and_preprocess_data('data/heart_failure.csv', fit=True)
```

#### Federated Learning Workflow

```python
from utils.preprocessing import HeartFailurePreprocessor, create_preprocessing_pipeline

# On central server: fit and save preprocessor
preprocessor = create_preprocessing_pipeline()
preprocessor.fit(global_training_data)
preprocessor.save('preprocessor.pkl')

# On federated client: load and use preprocessor
preprocessor = HeartFailurePreprocessor.load('preprocessor.pkl')
X_client, y_client = preprocessor.transform(client_data)
```

### Testing and Demo

Run the test suite to validate the preprocessing pipeline:

```bash
python test_preprocessing.py
```

Run the demo script to see usage examples:

```bash
python demo_preprocessing.py
```

## Model Architectures

The repository includes three FL-friendly model architectures for heart failure prediction:

### Available Models

1. **LSTM Classifier (PRIMARY)** ⭐
   - Shallow LSTM-based architecture (5,793 parameters)
   - Designed as the PRIMARY model for federated training
   - Best balance of performance and FL efficiency

2. **Temporal Convolutional Network (TCN)**
   - Lightweight causal convolution architecture (961 parameters)
   - Comparative model for benchmarking
   - Minimal parameter count for maximum FL efficiency

3. **Lightweight Transformer**
   - Simple attention-based architecture (1,497 parameters)
   - Comparative model for benchmarking
   - Modern architecture with residual connections

### Key Features

- **FL-friendly**: All models use shallow architectures (< 10K parameters)
- **DP-compatible**: All models support differential privacy training
- **Edge-optimized**: Suitable for deployment on edge devices
- **Well-documented**: Comprehensive documentation and examples

### Usage

```python
from models import get_primary_model
from utils.preprocessing import load_and_preprocess_data
import numpy as np

# Load and preprocess data
X, y, preprocessor = load_and_preprocess_data('data/heart_failure.csv', fit=True)

# Reshape for model input (add sequence dimension)
X_reshaped = X.reshape(-1, 1, 12)

# Get primary model (LSTM)
model = get_primary_model(input_shape=(1, 12))

# Model is ready for federated training
# model.fit(X_reshaped, y, ...)
```

### Testing and Demo

Run the test suite to validate all models:

```bash
python test_models.py
```

Run the demo script to see usage examples:

```bash
python demo_models.py
```

For detailed documentation, see [models/README.md](models/README.md).

## Federated Learning Clients

The repository includes Flower federated learning client implementations for privacy-preserving training across hospitals.

### Features

- **Privacy-Preserving**: Raw patient data never leaves the client
- **LSTM Model**: Uses the PRIMARY TensorFlow LSTM model
- **Non-IID Data**: Supports non-IID hospital datasets
- **Configurable**: Reusable client with configurable parameters
- **Secure**: No patient-level data logging, only aggregated metrics

### Usage

#### Basic Usage

```python
from federated import create_flower_client
from utils.client_partitioning import partition_for_federated_clients
from utils.preprocessing import create_preprocessing_pipeline
import pandas as pd

# Partition data for federated clients
client_datasets = partition_for_federated_clients(
    data_path='data/heart_failure.csv',
    n_clients=5,
    random_seed=42
)

# Create and fit preprocessing pipeline
data = pd.read_csv('data/heart_failure.csv')
preprocessor = create_preprocessing_pipeline()
preprocessor.fit(data)

# Create Flower client for a hospital
client = create_flower_client(
    client_data=client_datasets[0],
    preprocessor=preprocessor,
    val_split=0.2,
    epochs_per_round=5,
    batch_size=32,
    client_id="hospital_0"
)
```

#### Federated Training Workflow

```python
# Get initial model weights
params = client.get_parameters(config={})

# Train locally
updated_params, n_samples, metrics = client.fit(params, config={})

# Evaluate model
loss, n_samples, eval_metrics = client.evaluate(updated_params, config={})
```

### Testing and Demo

Run the test suite to validate federated client functionality:

```bash
python test_federated_client.py
```

Run the demo script to see federated learning in action:

```bash
python demo_federated_client.py
```

## Project Structure

```
UROP-B2-1/
├── data/
│   └── heart_failure.csv           # Heart failure clinical records dataset
├── utils/
│   ├── __init__.py                 # Package initialization
│   ├── preprocessing.py            # Preprocessing pipeline implementation
│   ├── data_sampling.py            # Dataset sampling module
│   └── client_partitioning.py      # Non-IID data partitioning for FL
├── models/
│   ├── __init__.py                 # Model registry and factory functions
│   ├── lstm_classifier.py          # LSTM Classifier (PRIMARY)
│   ├── tcn_classifier.py           # TCN Classifier
│   ├── transformer_classifier.py   # Transformer Classifier
│   └── README.md                   # Model documentation
├── federated/
│   ├── __init__.py                 # Federated learning module initialization
│   └── client.py                   # Flower federated client implementation
├── reports/
│   ├── data_profile.md             # Dataset profiling report
│   ├── sampling_summary.md         # Sampling operation report
│   └── client_partition_summary.md # Client partitioning report
├── validate_dataset.py             # Repository and dataset validation script
├── test_preprocessing.py           # Preprocessing pipeline test suite
├── test_sampling.py                # Dataset sampling test suite
├── test_models.py                  # Model architectures test suite
├── test_client_partitioning.py     # Client partitioning test suite
├── test_federated_client.py        # Federated client test suite
├── demo_preprocessing.py           # Preprocessing usage demonstration
├── demo_sampling.py                # Dataset sampling usage demonstration
├── demo_models.py                  # Model architectures demonstration
├── demo_client_partitioning.py     # Client partitioning demonstration
├── demo_federated_client.py        # Federated client demonstration
├── generate_data_profile.py        # Dataset profiling script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Notes

- Do NOT modify the dataset directly
- Do NOT create derived datasets without proper documentation
- The validation script only checks existence and readability, it does not perform preprocessing or training
- The preprocessing pipeline is designed for federated learning and must be used consistently across all clients

## Privacy and Security

This repository implements privacy-preserving federated learning with the following guarantees:

- **Data Privacy**: Raw patient data never leaves the client device
- **Weight-Only Sharing**: Only model weights are transmitted to the server
- **No Patient Logging**: Patient-level data is never logged, only aggregated metrics
- **Secure Training**: Local training on each hospital's own data
- **Non-IID Support**: Handles realistic non-IID data distributions across hospitals

For production deployments, consider additional security measures such as:
- Differential privacy during training
- Secure aggregation protocols
- Encrypted communication channels
- Access control and authentication