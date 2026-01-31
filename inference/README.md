# Inference Pipeline

This module provides inference capabilities for the trained federated learning model.

## Features

- Load trained global federated model
- Load shared preprocessing pipeline
- Accept new patient input data (CSV)
- Apply preprocessing and run inference
- Output predicted disease label and confidence score

## Constraints

- **NO training** occurs during inference
- **NO patient data** is stored
- Uses trained federated model artifacts
- All processing is done in-memory only

## Usage

### Python API

```python
from inference import InferencePipeline
import pandas as pd

# Initialize pipeline
pipeline = InferencePipeline()

# Load trained model
pipeline.load_model_from_history(
    history_path='logs/training_history.json',
    data_path='data/heart_failure.csv'
)

# Predict for a single patient
patient_data = pd.DataFrame([{
    'age': 75,
    'anaemia': 1,
    'creatinine_phosphokinase': 582,
    'diabetes': 0,
    'ejection_fraction': 20,
    'high_blood_pressure': 1,
    'platelets': 265000,
    'serum_creatinine': 1.9,
    'serum_sodium': 130,
    'sex': 1,
    'smoking': 0,
    'time': 4
}])

prediction, confidence = pipeline.predict(patient_data)
print(f"Prediction: {prediction}")  # "Yes" or "No"
print(f"Confidence: {confidence * 100:.2f}%")
```

### Batch Prediction from CSV

```python
from inference import predict_from_csv

# Make predictions for multiple patients
results = predict_from_csv('new_patients.csv')
print(results[['prediction', 'confidence']])
```

## Demo Script

Run the end-to-end demo:

```bash
python demo_inference.py
```

This demonstrates:
1. Creating sample patient data
2. Saving to CSV
3. Loading inference pipeline
4. Making predictions
5. Displaying results

## API Reference

### InferencePipeline

Main class for inference operations.

#### Methods

- `__init__()`: Initialize the inference pipeline
- `load_model_from_history(history_path, data_path)`: Load trained model
- `predict(patient_data)`: Predict for a single patient
- `predict_batch(patient_data)`: Predict for multiple patients

#### Returns

- `prediction`: "Yes" or "No" for DEATH_EVENT
- `confidence`: Probability score (0-1)

## Input Data Format

CSV file with the following columns (no DEATH_EVENT column):

- `age`: Patient age
- `anaemia`: 0 or 1
- `creatinine_phosphokinase`: CPK enzyme level
- `diabetes`: 0 or 1
- `ejection_fraction`: Percentage (0-100)
- `high_blood_pressure`: 0 or 1
- `platelets`: Platelet count
- `serum_creatinine`: Creatinine level
- `serum_sodium`: Sodium level
- `sex`: 0 (female) or 1 (male)
- `smoking`: 0 or 1
- `time`: Follow-up period (days)

## Security & Privacy

- ✓ No training occurs during inference
- ✓ No patient data is stored permanently
- ✓ All processing is in-memory only
- ✓ Uses pre-trained model artifacts
- ✓ Preprocessing pipeline ensures consistency

## Disclaimer

**⚠️ IMPORTANT**: This system is for research purposes only and is NOT a medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.
