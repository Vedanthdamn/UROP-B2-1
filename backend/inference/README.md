# PDF-Based Inference Pipeline

## Overview

This directory contains the **offline inference pipeline** for heart disease prediction from PDF blood reports.

**IMPORTANT**: This is **completely separate** from the federated learning training pipeline.

## Architecture Separation

```
Training (Federated Learning)     Inference (This Module)
         ↓                                ↓
   Train FL Model              →      Load Model
         ↓                                ↓
  Save global_model.h5           PDF → Extract → Predict
```

## Files

- `pdf_inference.py` - Main inference script
- `global_model.h5` - Pre-trained model (NOT in git, must be created)
- `__init__.py` - Python module file

## Setup

1. **Install dependencies**:
```bash
pip install pdfplumber pytesseract Pillow tensorflow

# For OCR support, install tesseract:
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

2. **Prepare the model**:
   - Train the federated model using the main training pipeline
   - Save the trained global model to `backend/inference/global_model.h5`

3. **Test the pipeline**:
```bash
python backend/inference/pdf_inference.py path/to/report.pdf
```

## Usage

### Basic Usage

```bash
python backend/inference/pdf_inference.py sample_report.pdf
```

### Custom Model Path

```bash
python backend/inference/pdf_inference.py report.pdf --model /path/to/model.h5
```

## What This Does

1. **PDF Text Extraction**:
   - Extracts text from digital PDFs using pdfplumber
   - Automatically detects scanned PDFs and applies OCR with pytesseract

2. **Rule-Based Value Extraction**:
   - Uses regex patterns to extract medical values
   - Supported fields: age, sex, blood pressure, cholesterol, fasting blood sugar, max heart rate
   - NO machine learning for extraction - only explicit patterns

3. **Feature Mapping**:
   - Maps to Heart Disease UCI schema (13 features)
   - Missing values filled with documented defaults

4. **Prediction**:
   - Binary classification: Disease / No Disease
   - Outputs probability and label

## Expected PDF Format

The pipeline looks for the following patterns in the PDF:

```
Age: 65 years
Sex: Male
Blood Pressure: 140/90 mmHg
Cholesterol: 250 mg/dL
Fasting Blood Sugar: 130 mg/dL
Max Heart Rate: 142 bpm
```

## Limitations

- **Fixed format**: Expects specific text patterns
- **Limited fields**: Only 6 extracted, rest use defaults
- **No semantic understanding**: Uses simple regex, not NLP
- **Not clinically validated**: Research demo only

## Important Notes

### Research Demo Only

⚠️ **This is NOT a medical device**
- Not validated for clinical use
- Not FDA-approved
- Not a substitute for professional medical advice
- For research and educational purposes only

### Architectural Notes

- OCR/NLP are **preprocessing steps**, NOT part of federated learning
- The FL training never sees PDFs - only structured data
- This module runs **completely offline** with no network communication
- No integration with Flower or federated training rounds

## Troubleshooting

### "Model not found" error

Make sure you have:
1. Trained the federated model
2. Saved it to `backend/inference/global_model.h5`
3. Used the correct file path

### "No text extracted" error

- Check if the PDF is corrupted
- Ensure tesseract is installed for OCR support
- Verify the PDF contains readable text

### "TesseractNotFoundError"

Install tesseract-ocr:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Example Output

```
================================================================================
PDF INFERENCE PIPELINE
================================================================================
Processing: sample_report.pdf

2024-01-31 12:00:00 - INFO - Loading model from: backend/inference/global_model.h5
2024-01-31 12:00:01 - INFO - Model loaded successfully!
2024-01-31 12:00:01 - INFO - Extracting text from PDF: sample_report.pdf
2024-01-31 12:00:02 - INFO - Extracted 6 values from PDF

2024-01-31 12:00:02 - INFO - Mapping extracted values to UCI Heart Disease schema...
2024-01-31 12:00:02 - INFO -   age: 65.0 (extracted)
2024-01-31 12:00:02 - INFO -   sex: 1.0 (extracted)
2024-01-31 12:00:02 - INFO -   trestbps: 140.0 (extracted)
...

================================================================================
RESULTS
================================================================================
Prediction: Disease
Probability: 72.34%

⚠️  DISCLAIMER: This is a research demonstration, NOT a medical device.
   Always consult qualified healthcare professionals for medical advice.
================================================================================
```

## Contact

For questions about this inference pipeline, please refer to the main repository README.
