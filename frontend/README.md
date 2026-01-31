# Frontend Web Interface

Simple web interface for heart failure prediction using the trained federated learning model.

## Features

- Simple, intuitive web interface
- File upload with "+" button
- Drag-and-drop CSV upload
- Display predicted disease (DEATH_EVENT: Yes/No)
- Display confidence score (probability)
- Visual risk indicators
- Research disclaimer

## Constraints

- **NO training** occurs
- **NO patient data** is stored
- All processing is in-memory only
- Uses trained federated model

## Running the Frontend

### Start the Web Server

```bash
python -m flask --app frontend.app run --host=0.0.0.0 --port=5000
```

Or directly:

```bash
cd frontend
python app.py
```

### Access the Interface

Open your browser and navigate to:

```
http://localhost:5000
```

## Using the Interface

1. **Upload CSV File**
   - Click the "+" upload button
   - Or drag and drop a CSV file
   - File must be in CSV format

2. **View Results**
   - Prediction: "Risk of Death Event: HIGH" or "Risk of Death Event: LOW"
   - Confidence Score: Percentage (0-100%)
   - Visual progress bar showing confidence

3. **Multiple Patients**
   - Upload CSV with multiple rows
   - Each patient gets a separate result card

## Input CSV Format

Your CSV file should have the following columns (no DEATH_EVENT column):

```csv
age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time
75,1,582,0,20,1,265000,1.9,130,1,0,4
45,0,120,0,60,0,300000,1.0,140,0,0,100
```

### Column Descriptions

- `age`: Patient age
- `anaemia`: 0 (no) or 1 (yes)
- `creatinine_phosphokinase`: CPK enzyme level (mcg/L)
- `diabetes`: 0 (no) or 1 (yes)
- `ejection_fraction`: Blood ejection percentage (0-100)
- `high_blood_pressure`: 0 (no) or 1 (yes)
- `platelets`: Platelet count (kiloplatelets/mL)
- `serum_creatinine`: Creatinine level (mg/dL)
- `serum_sodium`: Sodium level (mEq/L)
- `sex`: 0 (female) or 1 (male)
- `smoking`: 0 (no) or 1 (yes)
- `time`: Follow-up period (days)

## API Endpoints

### `GET /`
Main interface page

### `POST /predict`
Make predictions

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: CSV file

**Response:**
```json
{
  "success": true,
  "num_patients": 3,
  "predictions": [
    {
      "patient_id": 1,
      "prediction": "Yes",
      "confidence": 85.50,
      "confidence_raw": 0.8550
    }
  ]
}
```

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "pipeline_loaded": true
}
```

## Development

### Requirements

Install dependencies:

```bash
pip install flask>=2.0.0
```

### Configuration

The app can be configured by passing a config dictionary to `create_app()`:

```python
from frontend import create_app

config = {
    'SECRET_KEY': 'your-secret-key',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB
}

app = create_app(config)
app.run()
```

## Security & Privacy

- ✓ No data storage - all processing in-memory
- ✓ No training occurs
- ✓ Files are processed and discarded
- ✓ Uses pre-trained model only
- ✓ Session-based, no persistent storage

## Disclaimer

**⚠️ RESEARCH DISCLAIMER**

This system is for research purposes only and is NOT a medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## Troubleshooting

### Model Loading Fails

Ensure the following files exist:
- `logs/training_history.json`
- `data/heart_failure.csv`

### Port Already in Use

Change the port:

```bash
python -m flask --app frontend.app run --port=5001
```

### CSV Upload Fails

Ensure CSV has:
- Correct column names
- No DEATH_EVENT column (if present, it will be removed automatically)
- Numeric values for all fields
