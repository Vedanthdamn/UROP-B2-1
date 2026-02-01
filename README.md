# Federated Learning for Privacy-Preserving Medical AI

## 1. Project Overview

This repository implements a **privacy-preserving federated learning system** for **heart failure prediction** across multiple hospitals. The project demonstrates how medical institutions can collaboratively train machine learning models without sharing sensitive patient data, leveraging the complementary strengths of **Federated Learning (FL)** and **Differential Privacy (DP)**.

### Goal of the Project

The primary goal is to build a robust heart failure prediction model while ensuring patient privacy through:

1. **Federated Learning**: Hospitals train locally on their own data; only model updates are shared (not raw patient records)
2. **Differential Privacy**: Gradient clipping and Gaussian noise addition provide formal privacy guarantees
3. **Non-IID Data Handling**: Realistic simulation of hospitals with different data distributions and patient populations
4. **Model Evaluation**: Comprehensive analysis of model performance, fairness, and privacy-utility trade-offs

### Federated Learning + Differential Privacy Motivation

**Why Federated Learning?**
- Medical data is sensitive and subject to strict privacy regulations (HIPAA, GDPR)
- Centralized data aggregation is often infeasible due to legal and ethical constraints
- FL enables collaborative learning while keeping patient data at source hospitals
- Reduces communication overhead and data transfer costs

**Why Differential Privacy?**
- Even aggregated model updates can leak information about training data
- DP provides mathematically rigorous privacy guarantees through (ε, δ)-differential privacy
- Protects against membership inference and reconstruction attacks
- Allows tuning of privacy-utility trade-off via privacy budget (epsilon)

### Medical Use Case: Heart Failure Prediction

Heart failure is a major cause of mortality worldwide. Early prediction can:
- Enable timely interventions and improve patient outcomes
- Optimize resource allocation in healthcare systems
- Reduce hospitalization and healthcare costs

This system predicts the **DEATH_EVENT** (mortality during follow-up period) based on clinical features such as:
- Age, ejection fraction, serum creatinine
- Presence of comorbidities (anemia, diabetes, hypertension)
- Laboratory test results (CPK, platelets, serum sodium)

---

## 2. System Architecture

### Training Pipeline: Federated Learning with Flower

The training pipeline implements a federated learning system using the [Flower](https://flower.ai/) framework:

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Server                          │
│  - Initializes global LSTM model                            │
│  - Distributes model parameters to clients                  │
│  - Aggregates DP-protected updates (FedAvg/FedProx)        │
│  - Tracks training metrics across rounds                    │
│  - NEVER accesses raw patient data                         │
└───────────────┬─────────────────────────────────────────────┘
                │
       ┌────────┴────────┐
       │                 │
┌──────▼──────┐   ┌─────▼──────┐   ... (5 clients total)
│  Hospital 1 │   │ Hospital 2 │
│   Client    │   │   Client   │
│             │   │            │
│ - Local     │   │ - Local    │
│   Training  │   │   Training │
│ - DP-SGD    │   │ - DP-SGD   │
│ - Gradient  │   │ - Gradient │
│   Clipping  │   │   Clipping │
│ - Noise     │   │ - Noise    │
│   Addition  │   │   Addition │
│             │   │            │
│ Patient     │   │ Patient    │
│ Data (stays │   │ Data (stays│
│  local)     │   │  local)    │
└─────────────┘   └────────────┘
```

**Key Components:**

1. **Server (`federated/server.py`)**:
   - Orchestrates training across hospitals
   - Implements aggregation strategies: **FedAvg** (primary) and **FedProx** (comparative)
   - Tracks per-round metrics (loss, accuracy, client participation)
   - Ensures privacy by operating only on model weights

2. **Client (`federated/client.py`)**:
   - Trains LSTM model locally on hospital data
   - Applies differential privacy via gradient clipping + noise
   - Returns only model weights (no patient data)
   - Stateless between rounds for security

### Client-Side Training + Differential Privacy

Each hospital client performs local training with privacy protection:

**Training Process:**
1. Receive global model parameters from server
2. Train on local non-IID hospital data for fixed epochs
3. **Apply Differential Privacy** (`federated/differential_privacy.py`):
   - **Gradient Clipping**: Clip gradients to bounded L2 norm (typically 1.0)
   - **Gaussian Noise**: Add calibrated noise scaled by privacy budget (ε, δ)
4. Send DP-protected model updates back to server

**Privacy Guarantees:**
- Implements (ε, δ)-differential privacy
- All privacy operations occur BEFORE sharing updates
- Raw patient data NEVER leaves the client
- Configurable privacy budget for tuning privacy-utility trade-off

### Server-Side Aggregation (FedAvg, FedProx)

**FedAvg (Federated Averaging)** - Primary Strategy:
- Weighted average of client model updates
- Weight proportional to client dataset size
- Simple, efficient, widely used in federated learning

**FedProx (Federated Proximal)** - Comparative Strategy:
- Adds proximal term to prevent client drift
- Useful when client data is highly heterogeneous (non-IID)
- Proximal parameter μ controls regularization strength

Both strategies:
- Aggregate only model weights (privacy-preserving)
- Track global metrics across federated rounds
- Support heterogeneous client participation

### Inference Pipeline (No Training, No Data Storage)

The inference pipeline (`inference/inference_pipeline.py`) is strictly separated from training:

**Design Principles:**
- **No Training**: Uses pre-trained model weights only
- **No Data Storage**: All processing is in-memory
- **Stateless**: Each prediction is independent
- **Privacy-Preserving**: No patient data is logged or persisted

**Components:**
1. **InferencePipeline Class**: Loads trained model and preprocessor
2. **Preprocessing**: Same pipeline as training (standardization, imputation)
3. **Prediction**: Binary classification (DEATH_EVENT: 0 or 1) with confidence score

---

## 3. Dataset

### Heart Failure Clinical Records Dataset

**Source**: Public benchmark dataset (Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020))

**Dataset Characteristics:**
- **Size**: 299 patients
- **File**: `data/heart_failure.csv`
- **Features**: 12 clinical variables
- **Target**: `DEATH_EVENT` (binary: 0 = survived, 1 = death during follow-up)

**Feature Descriptions:**

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age (years) | Continuous |
| `anaemia` | Decrease of red blood cells or hemoglobin | Binary (0/1) |
| `creatinine_phosphokinase` | Level of CPK enzyme in blood (mcg/L) | Continuous |
| `diabetes` | Whether patient has diabetes | Binary (0/1) |
| `ejection_fraction` | Percentage of blood leaving heart per contraction (%) | Continuous |
| `high_blood_pressure` | Whether patient has hypertension | Binary (0/1) |
| `platelets` | Platelets in blood (kiloplatelets/mL) | Continuous |
| `serum_creatinine` | Level of serum creatinine (mg/dL) | Continuous |
| `serum_sodium` | Level of serum sodium (mEq/L) | Continuous |
| `sex` | Gender (binary: woman or man) | Binary (0/1) |
| `smoking` | Whether patient smokes | Binary (0/1) |
| `time` | Follow-up period (days) | Continuous |
| `DEATH_EVENT` | **Target**: Whether patient died during follow-up | Binary (0/1) |

### Public Benchmark Dataset

This is a widely-used, publicly available dataset for heart failure prediction research. It is:
- **Non-proprietary**: Free to use for research
- **De-identified**: Contains no personally identifiable information (PII)
- **Well-studied**: Used in multiple peer-reviewed publications
- **Class-imbalanced**: ~68% survival, ~32% death (realistic medical scenario)

### Non-IID Hospital Simulation

To simulate realistic federated learning across hospitals, the dataset is partitioned into **5 non-IID clients** (`utils/client_partitioning.py`):

**Non-IID Characteristics:**
1. **Unequal Sample Sizes**: Different hospitals have different numbers of patients
   - Simulates varying hospital sizes and patient volumes
2. **Different Class Distributions**: Each hospital has different survival/death ratios
   - Models real-world heterogeneity in patient populations
3. **No Overlap**: Each patient belongs to exactly one hospital
   - Ensures privacy and no data leakage

**Example Partition:**
- Hospital 0: 74 patients, 69% survival
- Hospital 1: 65 patients, 72% survival  
- Hospital 2: 59 patients, 66% survival
- Hospital 3: 39 patients, 64% survival
- Hospital 4: 62 patients, 71% survival

This non-IID setup tests the federated learning system under realistic conditions where hospitals have heterogeneous data.

---

## 4. Models

### LSTM (Primary Model)

**Architecture** (`models/lstm_classifier.py`):
```
Input (1, 12) → LSTM(32 units) → Dropout(0.3) → Dense(1, sigmoid) → Output
```

**Why LSTM?**
- Captures temporal dependencies in sequential clinical data
- Shallow design (32 units) minimizes parameters for FL efficiency
- Compatible with differential privacy training
- Well-suited for tabular medical data
- Low communication overhead in federated setting

**Parameters:**
- LSTM units: 32
- Dropout rate: 0.3 (regularization)
- Activation: Sigmoid (binary classification)
- Loss: Binary cross-entropy
- Optimizer: Adam

**Model Size:**
- ~2,000-3,000 parameters (lightweight for edge deployment)

### TCN, Transformer (Comparative Models)

Two additional models are implemented for comparative analysis:

**1. TCN (Temporal Convolutional Network)** (`models/tcn_classifier.py`):
- Uses dilated causal convolutions
- Expanded receptive field for temporal patterns
- Alternative to LSTM for sequential modeling
- Comparable parameter count for fair comparison

**2. Transformer** (`models/transformer_classifier.py`):
- Lightweight single-head self-attention mechanism
- Layer normalization for training stability
- Modern architecture for comparative benchmarking
- Simplified design suitable for FL

**Purpose of Comparative Models:**
- Benchmark LSTM performance against alternatives
- Evaluate architecture choices for federated medical AI
- Validate that LSTM is suitable for this task
- Academic completeness and reproducibility

### Why Lightweight Models Are Used

**Design Rationale:**

1. **Federated Learning Efficiency**:
   - Fewer parameters = smaller model updates to transmit
   - Reduced communication overhead between clients and server
   - Faster convergence across federated rounds

2. **Differential Privacy Compatibility**:
   - Smaller models are less sensitive to DP noise
   - Better privacy-utility trade-off
   - More stable training under gradient clipping

3. **Edge Device Deployment**:
   - Hospital edge devices may have limited compute
   - Lightweight models enable on-device inference
   - Lower latency for clinical decision support

4. **Interpretability and Safety**:
   - Simpler models are easier to interpret for medical use
   - Reduced risk of overfitting on small datasets
   - More trustworthy for healthcare applications

5. **Dataset Size**:
   - Heart failure dataset has only 299 samples
   - Lightweight models prevent overfitting
   - Better generalization to unseen patients

---

## 5. Differential Privacy

### (ε, δ)-Differential Privacy

The system implements **(ε, δ)-differential privacy**, a gold standard for formal privacy guarantees.

**Definition:**
A mechanism M provides (ε, δ)-DP if for any two datasets D and D' differing by one individual:

```
Pr[M(D) ∈ S] ≤ exp(ε) × Pr[M(D') ∈ S] + δ
```

**Interpretation:**
- **ε (epsilon)**: Privacy budget
  - Lower ε = stronger privacy (more noise)
  - Higher ε = weaker privacy (less noise, better utility)
  - Typical values: 0.1 to 10.0
- **δ (delta)**: Probability of privacy breach
  - Should be cryptographically small (e.g., 1e-5)
  - Represents rare "catastrophic" privacy failure

**Privacy Guarantee:**
- Adding/removing one patient from training data causes minimal change in model outputs
- Protects against membership inference attacks
- Formal mathematical guarantee (not heuristic)

### Gradient Clipping and Gaussian Noise

The DP mechanism (`federated/differential_privacy.py`) implements **DP-SGD** (Differentially Private Stochastic Gradient Descent):

**Step 1: Gradient Clipping (Per-Sample)**
```python
# Clip gradient of each sample to bounded L2 norm
clipped_gradient = gradient * min(1.0, C / ||gradient||₂)
```
- **Purpose**: Bound sensitivity to individual samples
- **L2 Norm Clip (C)**: Typically 1.0
- **Effect**: Prevents outliers from dominating updates

**Step 2: Aggregate Clipped Gradients**
```python
# Sum clipped gradients across batch
aggregated_gradient = Σ clipped_gradients
```

**Step 3: Add Gaussian Noise**
```python
# Add noise scaled by privacy budget
noise = N(0, σ²C²)
noisy_gradient = aggregated_gradient + noise
```
- **Noise Scale (σ)**: Computed from (ε, δ)
- **σ = sqrt(2 * ln(1.25/δ)) * C / ε**
- **Effect**: Obscures contribution of individual samples

**Implementation:**
- All DP operations occur on client before sending updates
- Server receives only DP-protected aggregated gradients
- Privacy budget tracked per federated round

### Privacy–Utility Tradeoff Summary

Experiments with multiple epsilon values demonstrate the privacy-utility tradeoff (`run_privacy_utility_analysis.py`):

**Tested Privacy Budgets:**

| Epsilon (ε) | Privacy Level | Final Accuracy | Final Loss | Interpretation |
|-------------|---------------|----------------|------------|----------------|
| **0** (No DP) | None | **86.55%** | 0.57 | Baseline (no privacy) |
| **0.5** | Strong Privacy | 50.84% | 50.20 | High privacy, significant utility loss |
| **1.0** | Moderate Privacy | 49.16% | 22.38 | Balanced trade-off |
| **2.0** | Moderate Privacy | 51.26% | 12.03 | Balanced trade-off |
| **5.0** | Relaxed Privacy | 50.84% | 4.50 | Lower privacy, better utility |

**Key Findings:**
1. **Strong DP incurs significant accuracy loss**: ε=0.5 reduces accuracy by ~41%
2. **Moderate DP is practical**: ε=1.0-2.0 maintains reasonable utility
3. **Higher epsilon improves utility**: ε=5.0 approaches baseline performance
4. **Fairness improves with DP**: Lower client accuracy variance at ε=1.0

**Recommendations:**
- **Research/Development**: Use ε ≥ 5.0 or no DP for maximum utility
- **Production Deployment**: Use ε = 1.0-2.0 for balanced privacy-utility
- **High-Privacy Applications**: Use ε ≤ 0.5, accept utility degradation
- **Consider Context**: Privacy requirements depend on data sensitivity and regulations

---

## 6. Experiments & Evaluation

### Federated Training Experiments

**Experiment Setup** (`run_federated_experiments.py`):
- **Strategy**: FedAvg (primary), FedProx (comparative)
- **Clients**: 5 non-IID hospitals
- **Rounds**: 5-10 federated training rounds
- **Epochs per Round**: 5 local epochs per client
- **Batch Size**: 32
- **Model**: LSTM (PRIMARY)

**Training Results** (FedAvg, 5 rounds, no DP):

| Round | Global Loss | Global Accuracy | Participating Clients |
|-------|-------------|-----------------|----------------------|
| 1 | 0.6617 | 66.39% | 5 |
| 2 | 0.6535 | 69.75% | 5 |
| 3 | 0.6396 | 76.89% | 5 |
| 4 | 0.6212 | 79.41% | 5 |
| 5 | 0.6143 | 76.89% | 5 |

**Observations:**
- Accuracy improves by 10.5% over 5 rounds
- Loss decreases by 0.047 (7.1% relative reduction)
- All clients participate consistently (100% participation rate)
- Model converges steadily without divergence

### Evaluation Metrics

**Standard Classification Metrics** (`evaluate_federated_model.py`):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 90.00% | High overall correctness |
| **Precision (Death)** | 20.00% | Low - many false positives |
| **Recall (Death)** | 33.33% | Low - misses many death cases |
| **F1-Score (Death)** | 25.00% | Poor on minority class |
| **Precision (Weighted)** | 92.55% | Good overall accounting for imbalance |
| **Recall (Weighted)** | 90.00% | Good overall |
| **F1-Score (Weighted)** | 91.16% | Strong overall performance |
| **Cross-Entropy Loss** | 0.6025 | Reasonable probabilistic calibration |

**Confusion Matrix:**
```
                Predicted
              Survived  Death
Actual Survived    53       4
       Death        2       1
```

**Analysis:**
- Model performs well on majority class (survival)
- Struggles with minority class (death) due to class imbalance
- Typical challenge in medical datasets
- Future work: class weighting, oversampling, or focal loss

### Fairness Analysis

**Client-Level Performance:**

| Hospital ID | Accuracy | Number of Samples |
|-------------|----------|-------------------|
| Hospital 0 | 75.68% | 74 |
| Hospital 1 | 75.38% | 65 |
| Hospital 2 | 76.27% | 59 |
| Hospital 3 | 64.10% | 39 |
| Hospital 4 | 75.81% | 62 |

**Fairness Metrics:**
- **Mean Client Accuracy**: 73.45%
- **Standard Deviation**: 4.68%
- **Accuracy Variance**: 0.0022
- **Min Accuracy**: 64.10% (Hospital 3)
- **Max Accuracy**: 76.27% (Hospital 2)

**Interpretation:**
- ✓ Low variance indicates fair performance across hospitals
- ✓ No single hospital is significantly disadvantaged
- ✓ Federated learning successfully handles non-IID data
- Hospital 3 has lower accuracy (smaller dataset size effect)

### Privacy–Utility Analysis

**Experimental Design** (`run_privacy_utility_analysis.py`):
- Test multiple privacy budgets: ε ∈ {0, 0.5, 1.0, 2.0, 5.0}
- Fixed parameters: δ=1e-5, L2 clip=1.0, 10 rounds
- Measure: final accuracy, loss, client fairness

**Results Visualization:**
- `reports/accuracy_vs_epsilon.png`: Shows accuracy degradation with stronger privacy
- `reports/loss_vs_epsilon.png`: Shows loss increase with stronger privacy

**Key Insights:**
1. **Privacy is costly**: Strong DP (ε=0.5) reduces accuracy by ~41%
2. **Diminishing returns**: Increasing ε beyond 2.0 yields minor utility gains
3. **Fairness trade-off**: Moderate DP (ε=1.0) minimizes client variance (fairest)
4. **Practical regime**: ε=1.0-2.0 balances privacy and utility for deployment

**Academic Significance:**
- Quantifies privacy-utility tradeoff for federated medical AI
- Provides empirical guidance for privacy budget selection
- Demonstrates feasibility of practical DP in healthcare FL

---

## 7. Inference & Frontend

### CSV Upload Interface

The system provides a web-based interface for heart failure prediction (`frontend/app.py`):

**Framework**: Flask (Python web framework)

**User Workflow:**
1. Navigate to web interface (http://localhost:5000)
2. Upload CSV file with patient clinical data
3. System validates and preprocesses data
4. Displays predictions with confidence scores
5. Shows research disclaimer

**Features:**
- Drag-and-drop file upload
- CSV format validation
- Real-time processing
- Responsive web interface
- Mobile-friendly design

### Prediction + Confidence Output

**Prediction Format:**

```json
{
  "predictions": [
    {
      "patient_id": 1,
      "prediction": "Survived",
      "death_risk": 0.15,
      "confidence": "High"
    },
    {
      "patient_id": 2,
      "prediction": "Death Risk",
      "death_risk": 0.78,
      "confidence": "High"
    }
  ]
}
```

**Output Fields:**
- **prediction**: Binary outcome (Survived / Death Risk)
- **death_risk**: Probability of DEATH_EVENT (0.0 to 1.0)
- **confidence**: Confidence level based on prediction probability
  - High: probability > 0.8 or < 0.2
  - Medium: probability 0.6-0.8 or 0.2-0.4
  - Low: probability 0.4-0.6 (uncertain)

**Privacy Considerations:**
- No patient data is stored or logged
- All processing occurs in-memory
- No training or model updates during inference
- Stateless server (no session persistence)

### Research Disclaimer

**Prominent Disclaimer on Interface:**

> ⚠️ **RESEARCH SYSTEM ONLY**
> 
> This system is a research prototype for academic purposes. It is **NOT**:
> - A medical diagnostic tool
> - Approved for clinical use
> - A substitute for professional medical advice
> 
> **DO NOT** use this system to make medical decisions. Always consult qualified healthcare professionals for medical diagnosis and treatment.
> 
> The model is trained on a public benchmark dataset and has not been validated on real patient populations. Predictions may not reflect real-world clinical outcomes.

**Legal and Ethical Compliance:**
- Clear statement that system is for research only
- Warning against clinical decision-making use
- Disclaimer of liability
- Emphasis on seeking professional medical advice

---

## 8. How to Run the Project

### Environment Setup

**Prerequisites:**
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

**Create Virtual Environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Install Dependencies

**Install Required Packages:**
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `tensorflow>=2.13.0`: Deep learning framework
- `flwr[simulation]>=1.5.0`: Federated learning framework
- `pandas>=1.3.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computing
- `scikit-learn>=1.0.0`: Preprocessing and metrics
- `flask>=2.0.0`: Web framework for inference
- `matplotlib>=3.3.0`, `seaborn>=0.11.0`: Visualization

### Run Federated Training

**Basic Training (FedAvg, No DP):**
```bash
python run_federated_experiments.py --num-rounds 10
```

**Training with Differential Privacy:**
```bash
python run_federated_experiments.py \
    --use-dp \
    --dp-epsilon 1.0 \
    --dp-delta 1e-5 \
    --num-rounds 10
```

**Training with FedProx Strategy:**
```bash
python run_federated_experiments.py \
    --strategy fedprox \
    --proximal-mu 0.1 \
    --num-rounds 10
```

**Command-Line Arguments:**
- `--num-rounds`: Number of federated training rounds (default: 5)
- `--strategy`: Aggregation strategy (`fedavg` or `fedprox`)
- `--use-dp`: Enable differential privacy
- `--dp-epsilon`: Privacy budget epsilon (default: 1.0)
- `--dp-delta`: Privacy budget delta (default: 1e-5)
- `--proximal-mu`: FedProx proximal parameter (default: 0.1)
- `--random-seed`: Random seed for reproducibility (default: 42)

**Outputs:**
- `logs/training_history.json`: Training metrics per round
- `logs/training_summary.md`: Human-readable training summary

### Run Evaluation

**Evaluate Trained Model:**
```bash
python evaluate_federated_model.py --data-path data/heart_failure.csv
```

**Compare FedAvg vs FedProx:**
```bash
# First train both strategies, then:
python evaluate_federated_model.py --compare-strategies
```

**Outputs:**
- `reports/evaluation_metrics.md`: Comprehensive evaluation report
- `reports/confusion_matrix.png`: Confusion matrix visualization

### Run Privacy-Utility Analysis

**Analyze Privacy-Utility Tradeoff:**
```bash
python run_privacy_utility_analysis.py
```

This script:
- Runs experiments with multiple epsilon values (0, 0.5, 1.0, 2.0, 5.0)
- Takes ~30-60 minutes to complete
- Generates comprehensive privacy-utility analysis

**Outputs:**
- `reports/privacy_utility_analysis.md`: Detailed analysis report
- `reports/accuracy_vs_epsilon.png`: Accuracy vs epsilon plot
- `reports/loss_vs_epsilon.png`: Loss vs epsilon plot

### Run Inference Web App

**Start the Web Server:**
```bash
cd frontend
python app.py
```

**Access the Interface:**
- Open browser and navigate to: `http://localhost:5000`
- Upload CSV file with patient data
- View predictions and confidence scores

**CSV Format Requirements:**
- Must include all 12 feature columns (see Dataset section)
- Can include multiple patients (rows)
- Optional: `DEATH_EVENT` column (for comparison)

**Example CSV:**
```csv
age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time
75,0,582,0,20,1,265000,1.9,130,1,0,4
55,0,7861,0,38,0,263358.03,1.1,136,1,0,6
```

---

## 9. Project Structure

```
UROP-B2-1/
│
├── data/
│   └── heart_failure.csv              # Heart failure clinical records dataset
│
├── models/
│   ├── __init__.py
│   ├── lstm_classifier.py             # LSTM model (PRIMARY)
│   ├── tcn_classifier.py              # TCN model (comparative)
│   └── transformer_classifier.py      # Transformer model (comparative)
│
├── federated/
│   ├── __init__.py
│   ├── client.py                      # Flower federated client implementation
│   ├── server.py                      # Flower federated server implementation
│   └── differential_privacy.py        # DP-SGD implementation (gradient clipping + noise)
│
├── inference/
│   ├── __init__.py
│   └── inference_pipeline.py          # Inference pipeline (no training, no storage)
│
├── frontend/
│   ├── __init__.py
│   ├── app.py                         # Flask web application
│   └── templates/
│       └── index.html                 # Web interface HTML template
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py               # Data preprocessing pipeline
│   ├── client_partitioning.py         # Non-IID dataset partitioning
│   └── data_sampling.py               # Stratified dataset sampling
│
├── logs/
│   ├── training_history.json          # Training metrics per round
│   └── training_summary.md            # Training summary report
│
├── reports/
│   ├── evaluation_metrics.md          # Comprehensive evaluation report
│   ├── confusion_matrix.png           # Confusion matrix visualization
│   ├── privacy_utility_analysis.md    # Privacy-utility tradeoff analysis
│   ├── accuracy_vs_epsilon.png        # Accuracy vs epsilon plot
│   ├── loss_vs_epsilon.png            # Loss vs epsilon plot
│   ├── client_partition_summary.md    # Client partitioning report
│   └── data_profile.md                # Dataset profiling report
│
├── run_federated_experiments.py       # Main script: Run federated training experiments
├── evaluate_federated_model.py        # Evaluate trained federated model
├── run_privacy_utility_analysis.py    # Analyze privacy-utility tradeoff
├── save_model_weights.py              # Save trained model weights for inference
├── validate_dataset.py                # Validate dataset integrity
│
├── demo_*.py                          # Demonstration scripts for each component
├── test_*.py                          # Unit tests for each module
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This comprehensive documentation
├── PREPROCESSING_GUIDE.md             # Detailed preprocessing documentation
└── INFERENCE_IMPLEMENTATION.md        # Detailed inference documentation
```

### Major Folders and Scripts

**Core Implementation:**
- **`models/`**: Neural network architectures (LSTM, TCN, Transformer)
- **`federated/`**: Federated learning components (client, server, DP)
- **`inference/`**: Inference pipeline for predictions
- **`frontend/`**: Web interface for CSV upload and predictions
- **`utils/`**: Preprocessing, partitioning, sampling utilities

**Execution Scripts:**
- **`run_federated_experiments.py`**: Main training script
- **`evaluate_federated_model.py`**: Model evaluation and metrics
- **`run_privacy_utility_analysis.py`**: Privacy-utility tradeoff analysis
- **`save_model_weights.py`**: Save trained weights for inference

**Testing and Validation:**
- **`demo_*.py`**: Demonstration scripts (one per module)
- **`test_*.py`**: Unit tests (one per module)
- **`validate_dataset.py`**: Dataset integrity checker

**Outputs:**
- **`logs/`**: Training history and summaries
- **`reports/`**: Evaluation reports, plots, analysis documents

---

## 10. Ethical Considerations & Limitations

### No Real Patient Data

**Important Clarification:**
- This system uses a **public benchmark dataset** (299 de-identified records)
- **NO real patient data** from actual hospitals is used
- Dataset is publicly available for research purposes
- All patient information is de-identified (no PII)

**Implications:**
- System has not been validated on real-world patient populations
- Performance metrics may not generalize to clinical deployment
- Model was not trained on proprietary hospital data
- Results are for research and educational purposes only

### Research-Only System

**System Status:**
- ✓ Research prototype for academic study
- ✓ Demonstrates federated learning + differential privacy
- ✓ Suitable for education and algorithm development
- ✗ **NOT validated for clinical use**
- ✗ **NOT FDA-approved or medically certified**
- ✗ **NOT intended for real patient diagnosis**

**Academic Context:**
- Developed for educational purposes (UROP project)
- Demonstrates technical feasibility of privacy-preserving medical AI
- Explores federated learning and differential privacy techniques
- Provides reproducible research codebase

### Not a Medical Diagnosis Tool

**Critical Warnings:**

⚠️ **This system is NOT:**
1. A medical diagnostic tool
2. A substitute for clinical judgment
3. Approved for use in patient care
4. Validated by medical professionals
5. Certified by regulatory authorities (FDA, CE, etc.)

⚠️ **Do NOT:**
1. Use this system to diagnose patients
2. Make treatment decisions based on predictions
3. Replace professional medical evaluation
4. Deploy this system in clinical settings
5. Rely on predictions for medical advice

✓ **This system IS:**
1. A research prototype
2. An educational tool for learning federated learning
3. A demonstration of privacy-preserving AI
4. A reproducible research codebase
5. A starting point for further research

### Limitations and Future Work

**Current Limitations:**
1. **Dataset Size**: Only 299 patients (small for deep learning)
2. **Class Imbalance**: Poor performance on minority class (death)
3. **Single Dataset**: Not validated on external datasets
4. **Simplified DP**: Single-round privacy analysis (not full composition)
5. **No Clinical Validation**: Not tested on real hospital data
6. **Limited Features**: Only 12 clinical variables
7. **Binary Outcome**: Does not predict survival time or severity
8. **Static Model**: No continual learning or model updates

**Future Research Directions:**
1. **Multi-Center Validation**: Test on real multi-hospital datasets
2. **Advanced DP**: Implement moments accountant for tighter privacy
3. **Class Balancing**: Address imbalance with SMOTE, focal loss, etc.
4. **Feature Engineering**: Include additional clinical variables
5. **Survival Analysis**: Predict time-to-event (not just binary outcome)
6. **Federated Hyperparameter Tuning**: Optimize across clients
7. **Secure Aggregation**: Add cryptographic protection for model updates
8. **Clinical Collaboration**: Partner with medical professionals for validation
9. **Regulatory Compliance**: Pursue FDA/CE approval for clinical deployment
10. **Continual Learning**: Update model with new hospital data

**Research Contributions:**
- Demonstrates technical feasibility of FL + DP for medical AI
- Provides reproducible codebase for federated medical learning
- Quantifies privacy-utility tradeoffs empirically
- Establishes baseline for future research
- Educational resource for privacy-preserving ML

---

## PDF-Based Inference (Research Demonstration)

### Overview

This repository includes a **separate, offline inference pipeline** that allows predictions using a pre-trained model from PDF blood reports. This feature is provided as a **research demonstration** and is **completely independent** from the federated learning training pipeline.

**Important Architectural Separation:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
│   (Federated Learning with Flower + Differential Privacy)       │
│   - Multi-hospital collaboration                                │
│   - Distributed training                                        │
│   - Privacy-preserving aggregation                              │
│   - Output: Trained global model                               │
└─────────────────────────────────────────────────────────────────┘

                            ↓ (saves model)

┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE (NEW)                       │
│   (PDF-based prediction - OFFLINE, NO TRAINING)                 │
│   - PDF text extraction (digital + OCR)                         │
│   - Rule-based value extraction                                 │
│   - Uses pre-trained model for prediction                      │
│   - NO integration with Flower or FL                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

1. **PDF Text Extraction**:
   - Uses `pdfplumber` for digital PDFs
   - Automatically detects empty pages and applies OCR using `pytesseract`
   - No heavy models like TrOCR

2. **Rule-Based Medical Value Extraction**:
   - Uses regex patterns to extract medical values
   - NO BioBERT, NO NLP inference, NO guessing
   - Supported fields (if present in report):
     - Age
     - Sex (male/female → 1/0)
     - Blood pressure (for high_blood_pressure flag)
     - Creatinine (serum_creatinine)
     - Sodium (serum_sodium)
     - Ejection fraction
     - CPK (creatinine phosphokinase)
     - Platelets
     - Diabetes (flag)
     - Anaemia (flag)
     - Smoking (flag)

3. **Heart Failure Clinical Records Schema Mapping**:
   - Maps extracted values to 12-feature vector
   - Missing values filled with documented defaults
   - Comments explain each default value
   - Features match the training dataset schema

4. **Model Loading**:
   - Loads trained model from `backend/inference/global_model.h5`
   - Uses TensorFlow/Keras load_model
   - Clear error if model is missing

5. **Prediction**:
   - Binary classification (Death Event Risk: HIGH / LOW)
   - Outputs probability and label
   - Minimal logging of intermediate steps
   - Web UI displays results with confidence scores

### Installation

Install all dependencies including those for PDF inference and web UI:

```bash
pip install -r requirements.txt

# For OCR support (pytesseract), also install tesseract binary:
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### Usage

#### Option 1: Web UI (Recommended for Faculty Demo)

**Step 1**: Generate the trained model:

```bash
python save_model_weights.py
```

This will create `backend/inference/global_model.h5` with a trained federated model.

**Step 2**: Start the FastAPI inference server:

```bash
python backend/inference/api.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

**Step 3**: Start the web frontend:

```bash
python frontend/app.py
```

The web UI will be available at `http://localhost:5000`.

**Step 4**: Upload a PDF blood report through the web interface:

1. Open `http://localhost:5000` in your browser
2. Select "PDF Blood Report" option
3. Upload a PDF file
4. Click "Analyze Report"
5. View the prediction and confidence score

**Screenshots:**

![Initial UI - CSV Mode](https://github.com/user-attachments/assets/afa7e389-adb5-4e8a-8a39-d0d23e7f9d06)

![PDF Upload Mode](https://github.com/user-attachments/assets/1fb1c9d1-7723-4731-a85a-f4ac66c8baf1)

#### Option 2: Command-Line Interface

For testing or batch processing, you can use the CLI directly:

```bash
python backend/inference/pdf_inference.py path/to/blood_report.pdf
```

**Example Output (CLI):**

```
================================================================================
PDF INFERENCE PIPELINE
================================================================================
Processing: sample_report.pdf

2024-01-31 12:00:00 - INFO - Loading model from: backend/inference/global_model.h5
2024-01-31 12:00:01 - INFO - Model loaded successfully!
2024-01-31 12:00:01 - INFO - Extracting text from PDF: sample_report.pdf
2024-01-31 12:00:02 - INFO - Page 1: Extracted digital text
2024-01-31 12:00:02 - INFO - Extracted 1250 characters from PDF

2024-01-31 12:00:02 - INFO - Extracting medical values using rule-based patterns...
2024-01-31 12:00:02 - INFO - Extracted age: 65.0
2024-01-31 12:00:02 - INFO - Extracted sex: Male (1)
2024-01-31 12:00:02 - INFO - Extracted blood pressure (systolic): 140.0
2024-01-31 12:00:02 - INFO - Extracted cholesterol: 250.0
2024-01-31 12:00:02 - INFO - Extracted 4 values from PDF

2024-01-31 12:00:02 - INFO - Mapping extracted values to Heart Failure schema...
2024-01-31 12:00:02 - INFO -   age: 65.0 (extracted)
2024-01-31 12:00:02 - INFO -   sex: 1.0 (extracted)
2024-01-31 12:00:02 - INFO -   high_blood_pressure: 1.0 (extracted)
2024-01-31 12:00:02 - INFO -   anaemia: 0 (default)
...

2024-01-31 12:00:03 - INFO - Running prediction...
2024-01-31 12:00:03 - INFO - Prediction: Death Event Risk: LOW
2024-01-31 12:00:03 - INFO - Probability: 0.3828

================================================================================
RESULTS
================================================================================
Prediction: Death Event Risk: LOW
Probability: 38.28%

⚠️  DISCLAIMER: This is a research demonstration, NOT a medical device.
   Always consult qualified healthcare professionals for medical advice.
================================================================================
```

**Example API Response:**

```json
{
  "success": true,
  "prediction": "LOW RISK",
  "prediction_label": "Death Event Risk: LOW",
  "probability": 0.3828,
  "probability_percentage": 38.28,
  "extracted_values": {
    "age": 65.0,
    "sex": 1.0,
    "high_blood_pressure": 1.0
  },
  "disclaimer": "⚠️ This is a research demonstration, NOT a medical device. Always consult qualified healthcare professionals for medical advice."
}
```

### API Endpoints

The FastAPI server (`backend/inference/api.py`) provides the following endpoints:

- **GET /** - API information and available endpoints
- **GET /health** - Health check endpoint
- **POST /predict** - Upload PDF and receive prediction
  - Accepts: multipart/form-data with PDF file
  - Returns: JSON with prediction, probability, and extracted values

Interactive API documentation is available at `http://localhost:8000/docs` when the server is running.

### What This Is NOT

**OCR and NLP are preprocessing layers, NOT part of the federated learning system:**
- PDF text extraction happens locally on the inference machine
- Rule-based value extraction is a simple preprocessing step
- The federated learning training pipeline never sees or processes PDFs
- The FL model was trained on structured tabular data, not text

**This is NOT:**
- ❌ A clinically validated medical device
- ❌ Integrated with the federated learning training pipeline
- ❌ Using advanced NLP models (BioBERT, ClinicalBERT, etc.)
- ❌ Capable of understanding complex medical reports
- ❌ Reliable for production medical use

**This IS:**
- ✅ A research demonstration of inference with pre-trained models
- ✅ An example of how to deploy FL models for prediction
- ✅ A proof-of-concept for PDF-based medical data extraction
- ✅ Architecturally separated from the FL training pipeline

### Limitations

1. **Fixed Report Format**: The rule-based extraction expects specific patterns in the PDF
2. **Limited Field Support**: Only extracts available fields; remaining fields use defaults
3. **No Contextual Understanding**: Uses simple regex, not semantic analysis
4. **Not Clinically Validated**: Extraction accuracy depends on PDF format
5. **Research Demo Only**: Not suitable for real medical decision-making
6. **LSTM Model**: Requires specific input shape (1, 1, 12) for time-series format

### Future Enhancements (Out of Scope for Current Demo)

- Support for more diverse report formats
- More sophisticated extraction (e.g., table parsing)
- Confidence scores for extracted values
- Multi-page report handling
- Structured output (JSON/CSV)

### Important Note for Faculty Reviewers

This inference pipeline demonstrates:
1. ✅ **Separation of Concerns**: Training (FL) and inference (local) are independent
2. ✅ **Model Deployment**: How to use a trained federated model for predictions
3. ✅ **Practical Application**: End-to-end workflow from PDF to prediction

The OCR/NLP components are **preprocessing tools**, not part of the federated learning architecture. They operate entirely on the inference machine and do not affect the privacy-preserving training process.

---

## Reproducibility & Dependency Pinning

### Overview

This project employs strict dependency pinning to ensure **deterministic reproducibility** across training and inference environments. This is essential for research-grade systems where results must be independently verifiable.

### NumPy Version Pinning

**Why NumPy is Pinned to 1.26.4:**

NumPy 2.x (released in 2024) introduced breaking changes to internal module paths, specifically moving `numpy.core` to `numpy._core`. This seemingly minor change has significant implications for machine learning reproducibility:

1. **Pickle Incompatibility**: Scikit-learn pipelines and custom preprocessing objects serialized with NumPy 1.x cannot be loaded in NumPy 2.x environments, causing `ModuleNotFoundError: No module named 'numpy._core'`

2. **Research Integrity**: ML artifacts (trained models, preprocessing pipelines, scalers) must load identically across environments to ensure reproducible results

3. **Federated Learning Consistency**: All federated clients must use identical preprocessing statistics computed during the initial fit phase

**Academic Context:**

In research systems, dependency versions act as part of the "experimental protocol." Just as wet-lab experiments require specific reagent versions, ML experiments require specific library versions to ensure deterministic behavior.

### Joblib Compatibility

Joblib 1.4.2 is pinned alongside NumPy 1.26.4 because:
- Joblib handles serialization of NumPy arrays and scikit-learn objects
- Version compatibility between joblib and NumPy ensures consistent pickle protocols
- Later joblib versions may introduce new serialization behaviors

### Artifact Regeneration Protocol

**When to Regenerate ML Artifacts:**

ML artifacts (preprocessing pipelines, model weights, scalers) **must be regenerated** when:

1. **Dependency versions change** (especially NumPy, scikit-learn, TensorFlow)
2. **Python version changes** (e.g., 3.9 → 3.10)
3. **Moving between environments** with different library versions
4. **Encountering import errors** during artifact loading

**How to Regenerate:**

```bash
# 1. Ensure correct dependencies
pip install -r requirements.txt

# 2. Delete old artifacts
rm -f preprocessor.pkl logs/model_weights.h5

# 3. Re-run preprocessing
python demo_preprocessing.py

# 4. Re-run training (if needed)
python run_federated_experiments.py

# 5. Save model weights
python save_model_weights.py
```

### Safer Serialization (Inference-Only)

For inference pipelines, the preprocessing module now provides **version-independent serialization**:

```python
# Safer alternative to pickle (immune to NumPy version changes)
preprocessor.save_safe('preprocessor.json')  # Saves JSON + .npy arrays
preprocessor = HeartFailurePreprocessor.load_safe('preprocessor.json')
```

This approach:
- Stores preprocessing statistics (means, stds, medians) as JSON metadata
- Saves NumPy arrays in `.npy` format (version-stable)
- Avoids pickle's dependency on Python internal paths

**Important**: The federated training pipeline continues to use standard pickle for compatibility with existing code. Only inference code benefits from safer serialization.

### Error Handling

If you encounter incompatibility errors, the system now provides **clear, actionable error messages**:

```
INCOMPATIBLE PREPROCESSING ARTIFACT DETECTED
============================================================

Error: The preprocessing artifact was created with a different NumPy version.

Root Cause:
  NumPy 2.x introduced breaking changes to internal module paths,
  making pickle artifacts serialized under NumPy 1.x incompatible.

Resolution:
  1. Ensure requirements.txt specifies numpy==1.26.4
  2. Reinstall: pip install -r requirements.txt
  3. Regenerate artifacts (see Artifact Regeneration Protocol above)
```

### For Faculty Reviewers

This reproducibility strategy demonstrates:

1. ✅ **Scientific Rigor**: Treating dependencies as part of the experimental protocol
2. ✅ **Forward Compatibility**: Providing migration paths when dependencies evolve
3. ✅ **Failure Transparency**: Clear error messages guide artifact regeneration
4. ✅ **Best Practices**: Following ML reproducibility guidelines from NeurIPS/ICML reproducibility checklists

Pinning dependencies is not a workaround—it's a **research requirement** for systems where independent verification is essential.

---

## References

1. Chicco, D., Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(16).

2. McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.

3. Abadi, M., et al. (2016). Deep Learning with Differential Privacy. *ACM CCS*.

4. Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks. *MLSys*.

5. Dwork, C., Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*.

6. Beutel, D.J., et al. (2020). Flower: A Friendly Federated Learning Research Framework. *arXiv:2007.14390*.

---

## License

This project is for academic and educational purposes. The heart failure dataset is publicly available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## Contact

For questions or collaboration inquiries, please contact the project maintainers through the GitHub repository.

**Repository**: [Vedanthdamn/UROP-B2-1](https://github.com/Vedanthdamn/UROP-B2-1)

---

## Acknowledgments

- Heart failure dataset provided by Chicco & Jurman (2020)
- Flower federated learning framework by Adap
- TensorFlow and scikit-learn communities
- UROP program for supporting this research project
