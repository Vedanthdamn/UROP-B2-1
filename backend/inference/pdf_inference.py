#!/usr/bin/env python3
"""
PDF-Based Inference Pipeline for Heart Disease Prediction

RESEARCH DEMONSTRATION - NOT A MEDICAL DEVICE

This module provides offline inference by:
1. Extracting text from PDF blood reports (digital or scanned)
2. Extracting medical values using rule-based regex patterns
3. Mapping to Heart Disease UCI schema (13 features)
4. Running prediction using pre-trained federated model

IMPORTANT:
- This is SEPARATE from the federated learning training pipeline
- This is OFFLINE inference using a previously trained model
- NO integration with Flower or FL rounds
- OCR/NLP are preprocessing layers, NOT part of FL

Author: UROP-B2 Team
"""

import os
import sys
import re
import logging
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import pdfplumber
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFInference:
    """
    PDF-based inference pipeline for heart disease prediction.
    
    This class handles:
    - PDF text extraction (digital + OCR fallback)
    - Rule-based medical value extraction
    - Feature mapping to Heart Disease UCI schema
    - Model loading and prediction
    """
    
    # Default values for missing features (documented)
    # Based on typical healthy adult values and dataset statistics
    DEFAULTS = {
        'age': 60,              # Median age from dataset
        'sex': 1,               # Male (1) as default
        'cp': 0,                # Chest pain type: asymptomatic
        'trestbps': 130,        # Resting blood pressure: normal range
        'chol': 240,            # Cholesterol: borderline high
        'fbs': 0,               # Fasting blood sugar: < 120 mg/dl
        'restecg': 0,           # Resting ECG: normal
        'thalach': 150,         # Max heart rate: moderate
        'exang': 0,             # Exercise induced angina: no
        'oldpeak': 0.0,         # ST depression: none
        'slope': 1,             # Slope of peak exercise ST: flat
        'ca': 0,                # Number of major vessels: 0
        'thal': 2               # Thalassemia: normal
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize the PDF inference pipeline.
        
        Args:
            model_path: Path to trained model (.h5 file).
                       Default: backend/inference/global_model.h5
        """
        if model_path is None:
            # Default path relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'global_model.h5')
        
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        if not os.path.exists(self.model_path):
            error_msg = (
                f"Model not found at: {self.model_path}\n"
                f"Please ensure you have:\n"
                f"1. Trained the federated model\n"
                f"2. Saved the global model to this location\n"
                f"3. Used the correct path"
            )
            raise FileNotFoundError(error_msg)
        
        try:
            import tensorflow as tf
            logger.info(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber.
        
        If page text is empty (scanned PDF), automatically apply OCR
        using pytesseract as fallback.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        full_text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try digital text extraction first
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # Digital PDF with extractable text
                        full_text += text + "\n"
                        logger.info(f"Page {page_num}: Extracted digital text")
                    else:
                        # Empty text - likely scanned PDF, apply OCR
                        logger.info(f"Page {page_num}: No digital text found, applying OCR...")
                        ocr_text = self._ocr_page(page)
                        full_text += ocr_text + "\n"
                        
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text
    
    def _ocr_page(self, page) -> str:
        """
        Apply OCR to a PDF page using pytesseract.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            OCR text
        """
        try:
            import pytesseract
            from io import BytesIO
            
            # Convert page to image
            img = page.to_image(resolution=300)
            pil_image = img.original
            
            # Apply OCR
            text = pytesseract.image_to_string(pil_image)
            logger.info("OCR completed successfully")
            return text
            
        except ImportError:
            logger.warning("pytesseract not installed, skipping OCR")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
    
    def extract_medical_values(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extract medical values from text using rule-based regex patterns.
        
        NO BioBERT, NO NLP inference - only explicit pattern matching.
        
        Supported fields (if present in report):
        - age
        - sex (male/female → 1/0)
        - blood pressure (systolic)
        - cholesterol
        - fasting blood sugar
        - max heart rate
        
        Args:
            text: Extracted text from PDF
            
        Returns:
            Dictionary of extracted values (None if not found)
        """
        logger.info("Extracting medical values using rule-based patterns...")
        values = {}
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Extract Age
        age_patterns = [
            r'age[:\s]+(\d{1,3})',
            r'(\d{1,3})\s*(?:years|yrs|y\.o\.)',
            r'patient.*?(\d{1,3})\s*(?:year|yr)',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['age'] = float(match.group(1))
                logger.info(f"Extracted age: {values['age']}")
                break
        
        # Extract Sex
        # Use more specific patterns to avoid false positives
        male_match = re.search(r'\b(?:male|man)\b', text_lower)
        female_match = re.search(r'\b(?:female|woman)\b', text_lower)
        
        if male_match and not female_match:
            values['sex'] = 1.0  # Male
            logger.info(f"Extracted sex: Male (1)")
        elif female_match and not male_match:
            values['sex'] = 0.0  # Female
            logger.info(f"Extracted sex: Female (0)")
        elif male_match and female_match:
            # Both terms present - try to determine context or default to male
            # Count occurrences to determine which is more likely
            male_count = len(re.findall(r'\b(?:male|man)\b', text_lower))
            female_count = len(re.findall(r'\b(?:female|woman)\b', text_lower))
            if female_count > male_count:
                values['sex'] = 0.0  # Female
                logger.info(f"Extracted sex: Female (0) - based on frequency")
            else:
                values['sex'] = 1.0  # Male
                logger.info(f"Extracted sex: Male (1) - based on frequency")
        
        # Extract Blood Pressure (systolic)
        bp_patterns = [
            r'blood\s*pressure[:\s]+(\d{2,3})\s*/\s*\d{2,3}',
            r'bp[:\s]+(\d{2,3})\s*/\s*\d{2,3}',
            r'systolic[:\s]+(\d{2,3})',
            r'(\d{2,3})\s*/\s*\d{2,3}\s*mmhg',
        ]
        for pattern in bp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['trestbps'] = float(match.group(1))
                logger.info(f"Extracted blood pressure (systolic): {values['trestbps']}")
                break
        
        # Extract Cholesterol
        chol_patterns = [
            r'cholesterol[:\s]+(\d{2,4})',
            r'chol[:\s]+(\d{2,4})',
            r'total\s*cholesterol[:\s]+(\d{2,4})',
            r'tc[:\s]+(\d{2,4})',
        ]
        for pattern in chol_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['chol'] = float(match.group(1))
                logger.info(f"Extracted cholesterol: {values['chol']}")
                break
        
        # Extract Fasting Blood Sugar
        fbs_patterns = [
            r'fasting\s*blood\s*sugar[:\s]+(\d{2,4})',
            r'fbs[:\s]+(\d{2,4})',
            r'fasting\s*glucose[:\s]+(\d{2,4})',
            r'fpg[:\s]+(\d{2,4})',
        ]
        for pattern in fbs_patterns:
            match = re.search(pattern, text_lower)
            if match:
                fbs_value = float(match.group(1))
                # Convert to binary: > 120 mg/dl = 1, else 0
                values['fbs'] = 1.0 if fbs_value > 120 else 0.0
                logger.info(f"Extracted fasting blood sugar: {fbs_value} mg/dl -> {values['fbs']}")
                break
        
        # Extract Max Heart Rate
        hr_patterns = [
            r'max(?:imum)?\s*heart\s*rate[:\s]+(\d{2,3})',
            r'max\s*hr[:\s]+(\d{2,3})',
            r'heart\s*rate\s*max[:\s]+(\d{2,3})',
            r'thalach[:\s]+(\d{2,3})',
        ]
        for pattern in hr_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['thalach'] = float(match.group(1))
                logger.info(f"Extracted max heart rate: {values['thalach']}")
                break
        
        logger.info(f"Extracted {len(values)} values from PDF")
        return values
    
    def map_to_uci_schema(self, extracted_values: Dict[str, Optional[float]]) -> np.ndarray:
        """
        Map extracted values to Heart Disease UCI schema.
        
        Heart Disease UCI dataset has 13 features:
        1. age: Age in years
        2. sex: Sex (1 = male, 0 = female)
        3. cp: Chest pain type (0-3)
        4. trestbps: Resting blood pressure (mm Hg)
        5. chol: Serum cholesterol (mg/dl)
        6. fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
        7. restecg: Resting ECG results (0-2)
        8. thalach: Maximum heart rate achieved
        9. exang: Exercise induced angina (1 = yes, 0 = no)
        10. oldpeak: ST depression induced by exercise
        11. slope: Slope of peak exercise ST segment (0-2)
        12. ca: Number of major vessels colored by fluoroscopy (0-3)
        13. thal: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
        
        Missing values are filled with documented defaults from self.DEFAULTS.
        
        Args:
            extracted_values: Dictionary of extracted medical values
            
        Returns:
            numpy array of shape (1, 13) ready for model input
        """
        logger.info("Mapping extracted values to UCI Heart Disease schema...")
        
        # Create feature vector using extracted values or defaults
        features = []
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for feature_name in feature_names:
            if feature_name in extracted_values and extracted_values[feature_name] is not None:
                value = extracted_values[feature_name]
                features.append(value)
                logger.info(f"  {feature_name}: {value} (extracted)")
            else:
                value = self.DEFAULTS[feature_name]
                features.append(value)
                logger.info(f"  {feature_name}: {value} (default)")
        
        # Convert to numpy array with shape (1, 13)
        feature_array = np.array(features, dtype=np.float32).reshape(1, -1)
        
        logger.info(f"Feature vector shape: {feature_array.shape}")
        logger.info(f"Feature vector: {feature_array[0]}")
        
        return feature_array
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Run prediction using the trained model.
        
        Args:
            features: numpy array of shape (1, 13)
            
        Returns:
            Tuple of (label, probability)
            - label: "Disease" or "No Disease"
            - probability: Probability of disease (0-1)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        logger.info("Running prediction...")
        
        # Make prediction
        prediction_proba = self.model.predict(features, verbose=0)
        probability = float(prediction_proba[0][0])
        
        # Convert to label
        label = "Disease" if probability >= 0.5 else "No Disease"
        
        logger.info(f"Prediction: {label}")
        logger.info(f"Probability: {probability:.4f}")
        
        return label, probability
    
    def process_pdf(self, pdf_path: str) -> Tuple[str, float, Dict]:
        """
        Complete pipeline: PDF → prediction.
        
        Args:
            pdf_path: Path to PDF blood report
            
        Returns:
            Tuple of (label, probability, extracted_values)
        """
        logger.info("=" * 80)
        logger.info("PDF INFERENCE PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Processing: {pdf_path}")
        logger.info("")
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        logger.info("")
        
        # Step 2: Extract medical values
        extracted_values = self.extract_medical_values(text)
        logger.info("")
        
        # Step 3: Map to UCI schema
        features = self.map_to_uci_schema(extracted_values)
        logger.info("")
        
        # Step 4: Run prediction
        label, probability = self.predict(features)
        logger.info("")
        
        logger.info("=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)
        logger.info(f"Prediction: {label}")
        logger.info(f"Probability: {probability:.2%}")
        logger.info("")
        logger.info("⚠️  DISCLAIMER: This is a research demonstration, NOT a medical device.")
        logger.info("   Always consult qualified healthcare professionals for medical advice.")
        logger.info("=" * 80)
        
        return label, probability, extracted_values


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PDF-Based Inference for Heart Disease Prediction (Research Demo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backend/inference/pdf_inference.py report.pdf
  python backend/inference/pdf_inference.py /path/to/blood_report.pdf --model custom_model.h5

Important Notes:
  - This is a research demonstration, NOT a medical device
  - Inference is OFFLINE and separate from federated learning training
  - OCR/NLP are preprocessing layers, not part of the FL system
  - Fixed report format with limited field support
  - Not clinically validated
        """
    )
    
    parser.add_argument(
        'pdf_path',
        help='Path to PDF blood report'
    )
    
    parser.add_argument(
        '--model',
        default=None,
        help='Path to trained model (.h5 file). Default: backend/inference/global_model.h5'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize inference pipeline
        pipeline = PDFInference(model_path=args.model)
        
        # Process PDF
        label, probability, extracted_values = pipeline.process_pdf(args.pdf_path)
        
        # Exit with success
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
