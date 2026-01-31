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

## Federated Server

The repository now includes a Flower federated server for coordinating privacy-preserving training across multiple hospitals.

### Features

- **Global Model Management**: Initializes and distributes TensorFlow LSTM model
- **Aggregation Strategies**: Supports FedAvg (primary) and FedProx (comparative)
- **Metrics Tracking**: Logs per-round loss, accuracy, and participating clients
- **Privacy-Preserving**: Server never accesses raw patient data
- **DP-Aware**: Treats all client updates as DP-protected
- **Simulation Mode**: Built-in support for testing and development

### Usage

#### Running a Federated Training Session

Basic usage with FedAvg (default):

```bash
python demo_federated_training.py --num-clients 5 --num-rounds 10
```

With FedProx strategy:

```bash
python demo_federated_training.py --strategy fedprox --proximal-mu 0.1
```

With differential privacy:

```bash
python demo_federated_training.py --use-dp --dp-epsilon 1.0 --dp-delta 1e-5
```

Custom configuration:

```bash
python demo_federated_training.py \
  --num-clients 5 \
  --num-rounds 10 \
  --strategy fedavg \
  --use-dp \
  --dp-epsilon 1.0 \
  --dp-l2-norm-clip 1.0
```

#### Programmatic Usage

```python
from federated import create_federated_server, start_server_simulation
from federated import create_flower_client, create_dp_config
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

# Optional: Create DP configuration
dp_config = create_dp_config(
    epsilon=1.0,
    delta=1e-5,
    l2_norm_clip=1.0,
    enabled=True
)

# Define client factory function
def client_fn(cid: str):
    client_id = int(cid)
    return create_flower_client(
        client_data=client_datasets[client_id],
        preprocessor=preprocessor,
        client_id=f"hospital_{client_id}",
        dp_config=dp_config  # Optional
    )

# Run federated training simulation
history = start_server_simulation(
    client_fn=client_fn,
    num_clients=5,
    strategy="fedavg",  # or "fedprox"
    num_rounds=10
)
```

### Aggregation Strategies

#### FedAvg (Federated Averaging)

The primary aggregation strategy that computes weighted averages of client model updates:

```python
from federated import create_federated_server

server = create_federated_server(
    strategy="fedavg",
    num_rounds=10,
    min_clients=3
)
```

#### FedProx (Federated Proximal)

An alternative strategy that adds a proximal term to improve convergence with non-IID data:

```python
server = create_federated_server(
    strategy="fedprox",
    num_rounds=10,
    min_clients=3,
    proximal_mu=0.1  # Proximal term coefficient
)
```

### Metrics Tracking

The server tracks and logs the following metrics per round:

- **Global Loss**: Weighted average of client training losses
- **Global Accuracy**: Weighted average of client training accuracies
- **Number of Participating Clients**: Clients that completed training
- **Total Samples**: Sum of samples across all clients
- **DP Status**: Whether differential privacy is enabled

### Testing

Run the test suite to validate server functionality:

```bash
python test_federated_server.py
```

Run the demo script to see federated training in action:

```bash
python demo_federated_training.py
```

## Running Federated Training Experiments

The repository includes a comprehensive script for running end-to-end federated training experiments with full logging and result saving capabilities.

### Features

- **5 Non-IID Hospital Clients**: Realistic data distribution across hospitals
- **LSTM Model**: Uses the PRIMARY model for all experiments
- **Fixed Communication Rounds**: Configurable number of training rounds
- **Multiple Strategies**: Supports both FedAvg and FedProx
- **Comprehensive Logging**:
  - Per-round global accuracy
  - Per-round global loss
  - Client participation per round
  - Training history saved to `logs/training_history.json`
  - Training summary saved to `logs/training_summary.md`
- **Privacy-Preserving**: Optional differential privacy support

### Usage

#### Run with FedAvg (default):

```bash
python run_federated_experiments.py --num-rounds 10
```

#### Run with FedProx:

```bash
python run_federated_experiments.py --strategy fedprox --num-rounds 10 --proximal-mu 0.1
```

#### Run with Differential Privacy:

```bash
python run_federated_experiments.py --use-dp --dp-epsilon 1.0 --dp-delta 1e-5 --num-rounds 10
```

#### Custom Configuration:

```bash
python run_federated_experiments.py \
  --num-clients 5 \
  --num-rounds 10 \
  --strategy fedavg \
  --random-seed 42 \
  --output-dir logs
```

### Command-line Options

- `--num-clients`: Number of hospital clients (default: 5)
- `--num-rounds`: Number of federated training rounds (default: 10)
- `--strategy`: Aggregation strategy - `fedavg` or `fedprox` (default: fedavg)
- `--use-dp`: Enable differential privacy protection (flag)
- `--dp-epsilon`: Privacy budget epsilon (default: 1.0)
- `--dp-delta`: Privacy budget delta (default: 1e-5)
- `--dp-l2-norm-clip`: L2 norm clipping threshold (default: 1.0)
- `--proximal-mu`: Proximal term for FedProx (default: 0.1)
- `--data-path`: Path to dataset (default: data/heart_failure.csv)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Directory to save results (default: logs)

### Output Files

After running an experiment, results are saved to the `logs/` directory:

#### `logs/training_history.json`

Contains complete training metrics in JSON format:
- Experiment configuration
- Per-round metrics (loss, accuracy, client participation, total samples)
- Timestamp

Example:
```json
{
  "experiment_config": {
    "num_clients": 5,
    "num_rounds": 10,
    "strategy": "fedavg",
    "model": "LSTM (PRIMARY)",
    "data_partitioning": "Non-IID"
  },
  "rounds": [
    {
      "round": 1,
      "global_loss": 0.6776,
      "global_accuracy": 0.5588,
      "participating_clients": 5,
      "total_samples": 238
    }
  ]
}
```

#### `logs/training_summary.md`

A human-readable markdown report containing:
- Experiment configuration
- Per-round metrics in a table
- Training summary statistics (initial/final/improvement)
- Privacy guarantees
- Client participation summary

### Constraints

The script adheres to the following constraints:
- Does NOT modify model architecture
- Does NOT modify preprocessing
- Uses existing DP configuration when enabled
- Only logs aggregated metrics (no patient-level data)

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
│   ├── client.py                   # Flower federated client implementation
│   ├── server.py                   # Flower federated server implementation
│   └── differential_privacy.py     # Differential privacy implementation
├── reports/
│   ├── data_profile.md             # Dataset profiling report
│   ├── sampling_summary.md         # Sampling operation report
│   └── client_partition_summary.md # Client partitioning report
├── logs/
│   ├── training_history.json       # Federated training metrics (JSON)
│   └── training_summary.md         # Federated training summary (Markdown)
├── validate_dataset.py             # Repository and dataset validation script
├── test_preprocessing.py           # Preprocessing pipeline test suite
├── test_sampling.py                # Dataset sampling test suite
├── test_models.py                  # Model architectures test suite
├── test_client_partitioning.py     # Client partitioning test suite
├── test_federated_client.py        # Federated client test suite
├── test_federated_server.py        # Federated server test suite
├── test_differential_privacy.py    # Differential privacy test suite
├── demo_preprocessing.py           # Preprocessing usage demonstration
├── demo_sampling.py                # Dataset sampling usage demonstration
├── demo_models.py                  # Model architectures demonstration
├── demo_client_partitioning.py     # Client partitioning demonstration
├── demo_federated_client.py        # Federated client demonstration
├── demo_federated_training.py      # Full federated training session demonstration
├── demo_differential_privacy.py    # Differential privacy demonstration
├── run_federated_experiments.py    # End-to-end federated training experiments
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
- **Differential Privacy**: Optional (ε, δ)-DP protection for model updates

### Differential Privacy

The repository now supports differential privacy (DP) for federated learning, providing formal privacy guarantees for model updates:

#### Features

- **Gradient Clipping**: Bounds the sensitivity of model updates via L2 norm clipping
- **Gaussian Noise Addition**: Adds calibrated noise to provide (ε, δ)-differential privacy
- **Configurable Privacy Budget**: Tune epsilon and delta parameters to balance privacy and utility
- **Privacy Budget Tracking**: Logs all DP parameters for audit and compliance
- **Loss Monitoring**: Tracks training loss before and after DP application

#### Usage

```python
from federated import create_flower_client, create_dp_config
from utils.preprocessing import create_preprocessing_pipeline
from utils.client_partitioning import partition_for_federated_clients
import pandas as pd

# Partition data for federated clients
client_datasets = partition_for_federated_clients(
    data_path='data/heart_failure.csv',
    n_clients=5,
    random_seed=42
)

# Create and fit preprocessor
data = pd.read_csv('data/heart_failure.csv')
preprocessor = create_preprocessing_pipeline()
preprocessor.fit(data)

# Create DP configuration
dp_config = create_dp_config(
    epsilon=1.0,      # Privacy budget (lower = stronger privacy)
    delta=1e-5,       # Privacy budget (should be < 1/n_samples)
    l2_norm_clip=1.0, # Maximum L2 norm for gradient clipping
    enabled=True
)

# Create Flower client with DP
client = create_flower_client(
    client_data=client_datasets[0],
    preprocessor=preprocessor,
    val_split=0.2,
    epochs_per_round=5,
    batch_size=32,
    client_id="hospital_0",
    dp_config=dp_config  # Enable DP
)

# Train with DP protection
params = client.get_parameters(config={})
updated_params, n_samples, metrics = client.fit(params, config={})

# DP metrics are logged
print(f"DP epsilon: {metrics['dp_epsilon']}")
print(f"DP delta: {metrics['dp_delta']}")
print(f"Train loss (before DP): {metrics['train_loss']:.4f}")
print(f"Train loss (after DP): {metrics['train_loss_after_dp']:.4f}")
```

#### Privacy Parameters

- **epsilon (ε)**: Privacy budget parameter. Lower values = stronger privacy.
  - Typical values: 0.1 to 10.0
  - Recommendation: Start with ε=1.0 and adjust based on privacy requirements
  
- **delta (δ)**: Probability of privacy guarantee failure. Should be cryptographically small.
  - Typical values: 1e-5 to 1e-7
  - Recommendation: Set to 1/n_samples or smaller
  
- **l2_norm_clip**: Maximum L2 norm for gradient clipping. Controls sensitivity.
  - Typical values: 0.1 to 5.0
  - Recommendation: Start with 1.0 and adjust based on model updates

#### Testing and Demo

Run the test suite to validate DP functionality:

```bash
python test_differential_privacy.py
```

Run the demo script to see DP in action:

```bash
python demo_differential_privacy.py
```

#### Privacy Guarantees

When DP is enabled:
- Model updates satisfy (ε, δ)-differential privacy
- Gradient clipping is applied BEFORE model updates are sent to server
- Gaussian noise is calibrated based on sensitivity and privacy budget
- Raw gradients are NEVER exposed
- Privacy budget parameters are logged for audit

For production deployments, consider additional security measures such as:
- Secure aggregation protocols
- Encrypted communication channels
- Access control and authentication

## Model Evaluation

The repository includes a comprehensive evaluation script for trained federated learning models, supporting both single model evaluation and strategy comparison (FedAvg vs FedProx).

### Features

- **Standard Metrics**: Accuracy, precision, recall, F1-score
- **Cross-Entropy Loss**: Model confidence evaluation
- **Confusion Matrix**: Visual representation of classification performance
- **Client-Level Metrics**: Per-client accuracy for fairness analysis
- **Fairness Metrics**: Accuracy variance across clients
- **Strategy Comparison**: Compare FedAvg vs FedProx performance
- **Comprehensive Reports**: Markdown reports and visualizations

### Usage

#### Basic Evaluation

Evaluate a trained model using existing training history:

```bash
# Evaluate the model from training history
python evaluate_federated_model.py \
    --data-path data/heart_failure.csv \
    --history-path logs/training_history.json \
    --output-dir reports
```

#### Compare FedAvg vs FedProx

To compare two strategies, first train both models:

```bash
# Train with FedAvg
python run_federated_experiments.py \
    --strategy fedavg \
    --num-rounds 10 \
    --output-dir logs

# Rename output for comparison
mv logs/training_history.json logs/training_history_fedavg.json

# Train with FedProx
python run_federated_experiments.py \
    --strategy fedprox \
    --proximal-mu 0.1 \
    --num-rounds 10 \
    --output-dir logs

# Rename output for comparison
mv logs/training_history.json logs/training_history_fedprox.json

# Compare strategies
python evaluate_federated_model.py \
    --compare-strategies \
    --fedavg-history logs/training_history_fedavg.json \
    --fedprox-history logs/training_history_fedprox.json \
    --output-dir reports
```

### Evaluation Outputs

The evaluation script generates:

1. **`reports/evaluation_metrics.md`**: Comprehensive evaluation report including:
   - Experiment configuration
   - Standard classification metrics (accuracy, precision, recall, F1-score)
   - Weighted metrics for class imbalance handling
   - Class distribution analysis
   - Confusion matrix in text format
   - Client-level performance breakdown
   - Fairness metrics and interpretation
   - Training progress summary
   - Overall conclusions

2. **`reports/confusion_matrix.png`**: Visual confusion matrix showing:
   - True positives, true negatives, false positives, false negatives
   - Color-coded heatmap for easy interpretation

3. **`reports/evaluation_comparison.md`** (when comparing strategies):
   - Side-by-side comparison of FedAvg vs FedProx
   - Performance metrics for each strategy
   - Fairness comparison
   - Winner determination for each metric

### Understanding the Metrics

#### Standard Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that are identified
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Entropy Loss**: Measure of prediction confidence

#### Weighted Metrics
For datasets with class imbalance (like medical data), weighted metrics provide a more accurate assessment by accounting for the support (number of samples) of each class.

#### Fairness Metrics
- **Mean Client Accuracy**: Average accuracy across all clients
- **Standard Deviation**: Measure of consistency across clients
- **Accuracy Variance**: Quantifies performance disparity
- **Min/Max Client Accuracy**: Range of performance

**Interpretation**:
- Low variance (<0.01): Fair and equitable performance
- Moderate variance (0.01-0.05): Some performance disparity
- High variance (>0.05): Significant performance disparity

### Advanced Usage

#### Custom Parameters

```bash
python evaluate_federated_model.py \
    --data-path data/heart_failure.csv \
    --history-path logs/training_history.json \
    --output-dir custom_reports \
    --num-clients 5 \
    --random-seed 42
```

#### Programmatic Usage

```python
from evaluate_federated_model import evaluate_federated_model

# Run evaluation programmatically
evaluate_federated_model(
    data_path="data/heart_failure.csv",
    history_path="logs/training_history.json",
    output_dir="reports",
    num_clients=5,
    random_seed=42
)
```

### Important Notes

1. **No Retraining**: The evaluation script does NOT retrain the model. It recreates the trained model by simulating the federated training process.

2. **Preprocessing Consistency**: Uses the same preprocessing pipeline as training to ensure consistent evaluation.

3. **Class Imbalance**: Medical datasets often have class imbalance. The script provides both standard and weighted metrics for comprehensive assessment.

4. **Privacy Preservation**: Evaluation maintains privacy guarantees - only aggregated metrics are reported, no patient-level data is exposed.

### Example Output

After running evaluation, you'll see:

```
================================================================================
EVALUATION COMPLETED SUCCESSFULLY
Results saved to:
  - reports/evaluation_metrics.md
  - reports/confusion_matrix.png
================================================================================
```

The evaluation report provides actionable insights:
- Overall model performance on held-out test data
- Model quality assessment accounting for class imbalance
- Fairness analysis across different hospital clients
- Consistency of model performance across diverse datasets

## Complete Workflow

### End-to-End Federated Learning Pipeline

1. **Validate Dataset**
   ```bash
   python validate_dataset.py
   ```

2. **Run Federated Training**
   ```bash
   python run_federated_experiments.py --num-rounds 10 --strategy fedavg
   ```

3. **Evaluate Model**
   ```bash
   python evaluate_federated_model.py
   ```

4. **Review Results**
   - Check `logs/training_summary.md` for training progress
   - Check `reports/evaluation_metrics.md` for comprehensive evaluation
   - View `reports/confusion_matrix.png` for visual performance analysis
- Regular privacy audits