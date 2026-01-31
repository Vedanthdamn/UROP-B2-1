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
    PDF-based inference pipeline for heart failure prediction.
    
    This class handles:
    - PDF text extraction (digital + OCR fallback)
    - Rule-based medical value extraction
    - Feature mapping to Heart Failure Clinical Records schema
    - Model loading and prediction
    """
    
    # Default values for missing features (documented)
    # Based on typical healthy adult values and dataset statistics
    DEFAULTS = {
        'age': 60,                      # Median age from dataset
        'anaemia': 0,                   # No anaemia (0 = no, 1 = yes)
        'creatinine_phosphokinase': 250,  # CPK enzyme level (mcg/L)
        'diabetes': 0,                  # No diabetes (0 = no, 1 = yes)
        'ejection_fraction': 38,        # Heart ejection fraction percentage
        'high_blood_pressure': 0,       # No hypertension (0 = no, 1 = yes)
        'platelets': 263000,            # Platelet count (kiloplatelets/mL)
        'serum_creatinine': 1.1,        # Serum creatinine level (mg/dL)
        'serum_sodium': 137,            # Serum sodium level (mEq/L)
        'sex': 1,                       # Male (1) as default
        'smoking': 0,                   # Non-smoker (0 = no, 1 = yes)
        'time': 130                     # Follow-up period days
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
        - blood pressure (for high_blood_pressure flag)
        - creatinine
        - sodium
        - ejection fraction
        - CPK (creatinine phosphokinase)
        - platelets
        - diabetes (flag)
        - anaemia (flag)
        - smoking (flag)
        
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
        # Match 'female' first to avoid matching 'male' within 'female'
        female_match = re.search(r'\b(?:female|woman)\b', text_lower)
        # Use negative lookbehind to avoid matching 'male' within 'female'
        male_match = re.search(r'(?<!fe)\b(?:male|man)\b', text_lower)
        
        if male_match and not female_match:
            values['sex'] = 1.0  # Male
            logger.info(f"Extracted sex: Male (1)")
        elif female_match and not male_match:
            values['sex'] = 0.0  # Female
            logger.info(f"Extracted sex: Female (0)")
        elif male_match and female_match:
            # Both terms present - try to determine context or default to male
            # Count occurrences to determine which is more likely
            female_count = len(re.findall(r'\b(?:female|woman)\b', text_lower))
            male_count = len(re.findall(r'(?<!fe)\b(?:male|man)\b', text_lower))
            if female_count > male_count:
                values['sex'] = 0.0  # Female
                logger.info(f"Extracted sex: Female (0) - based on frequency")
            else:
                values['sex'] = 1.0  # Male
                logger.info(f"Extracted sex: Male (1) - based on frequency")
        
        # Extract Blood Pressure (for high_blood_pressure flag)
        bp_patterns = [
            r'blood\s*pressure[:\s]+(\d{2,3})\s*/\s*(\d{2,3})',
            r'bp[:\s]+(\d{2,3})\s*/\s*(\d{2,3})',
            r'systolic[:\s]+(\d{2,3})',
        ]
        for pattern in bp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                systolic = float(match.group(1))
                # High BP if systolic >= 140 or diastolic >= 90
                values['high_blood_pressure'] = 1.0 if systolic >= 140 else 0.0
                logger.info(f"Extracted blood pressure (systolic): {systolic} -> high_blood_pressure={values['high_blood_pressure']}")
                break
        
        # Extract Creatinine
        creat_patterns = [
            r'creatinine[:\s]+(\d+\.?\d*)',
            r'serum\s*creatinine[:\s]+(\d+\.?\d*)',
        ]
        for pattern in creat_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['serum_creatinine'] = float(match.group(1))
                logger.info(f"Extracted serum creatinine: {values['serum_creatinine']}")
                break
        
        # Extract Sodium
        sodium_patterns = [
            r'sodium[:\s]+(\d{2,3})',
            r'serum\s*sodium[:\s]+(\d{2,3})',
            r'na[:\s]+(\d{2,3})',
        ]
        for pattern in sodium_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['serum_sodium'] = float(match.group(1))
                logger.info(f"Extracted serum sodium: {values['serum_sodium']}")
                break
        
        # Extract Ejection Fraction
        ef_patterns = [
            r'ejection\s*fraction[:\s]+(\d{1,3})',
            r'ef[:\s]+(\d{1,3})(?:\%|percent)',
        ]
        for pattern in ef_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['ejection_fraction'] = float(match.group(1))
                logger.info(f"Extracted ejection fraction: {values['ejection_fraction']}")
                break
        
        # Extract CPK (Creatinine Phosphokinase)
        cpk_patterns = [
            r'cpk[:\s]+(\d{2,5})',
            r'creatinine\s*phosphokinase[:\s]+(\d{2,5})',
        ]
        for pattern in cpk_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['creatinine_phosphokinase'] = float(match.group(1))
                logger.info(f"Extracted CPK: {values['creatinine_phosphokinase']}")
                break
        
        # Extract Platelets
        platelet_patterns = [
            r'platelet[s]?[:\s]+(\d{3,7})',
            r'plt[:\s]+(\d{3,7})',
        ]
        for pattern in platelet_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['platelets'] = float(match.group(1))
                logger.info(f"Extracted platelets: {values['platelets']}")
                break
        
        # Extract Diabetes flag
        diabetes_patterns = [
            r'\b(?:diabetes|diabetic)\b',
        ]
        for pattern in diabetes_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['diabetes'] = 1.0
                logger.info(f"Detected diabetes: {values['diabetes']}")
                break
        
        # Extract Anaemia flag
        anaemia_patterns = [
            r'\b(?:an[ae]mia|an[ae]mic)\b',
        ]
        for pattern in anaemia_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['anaemia'] = 1.0
                logger.info(f"Detected anaemia: {values['anaemia']}")
                break
        
        # Extract Smoking flag
        smoking_patterns = [
            r'\b(?:smok(?:er|ing)|tobacco)\b',
        ]
        for pattern in smoking_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values['smoking'] = 1.0
                logger.info(f"Detected smoking: {values['smoking']}")
                break
        
        logger.info(f"Extracted {len(values)} values from PDF")
        return values
    
    def map_to_uci_schema(self, extracted_values: Dict[str, Optional[float]]) -> np.ndarray:
        """
        Map extracted values to Heart Failure Clinical Records schema.
        
        Heart Failure dataset has 12 features (excluding DEATH_EVENT target):
        1. age: Age in years
        2. anaemia: Decrease of red blood cells or hemoglobin (0 = no, 1 = yes)
        3. creatinine_phosphokinase: Level of CPK enzyme in blood (mcg/L)
        4. diabetes: If the patient has diabetes (0 = no, 1 = yes)
        5. ejection_fraction: Percentage of blood leaving the heart at each contraction
        6. high_blood_pressure: If the patient has hypertension (0 = no, 1 = yes)
        7. platelets: Platelet count in blood (kiloplatelets/mL)
        8. serum_creatinine: Level of serum creatinine in blood (mg/dL)
        9. serum_sodium: Level of serum sodium in blood (mEq/L)
        10. sex: Woman (0) or man (1)
        11. smoking: If the patient smokes (0 = no, 1 = yes)
        12. time: Follow-up period (days)
        
        Missing values are filled with documented defaults from self.DEFAULTS.
        
        Args:
            extracted_values: Dictionary of extracted medical values
            
        Returns:
            numpy array of shape (1, 1, 12) ready for model input (LSTM format)
        """
        logger.info("Mapping extracted values to Heart Failure schema...")
        
        # Create feature vector using extracted values or defaults
        features = []
        feature_names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                        'ejection_fraction', 'high_blood_pressure', 'platelets', 
                        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
        
        for feature_name in feature_names:
            if feature_name in extracted_values and extracted_values[feature_name] is not None:
                value = extracted_values[feature_name]
                features.append(value)
                logger.info(f"  {feature_name}: {value} (extracted)")
            else:
                value = self.DEFAULTS[feature_name]
                features.append(value)
                logger.info(f"  {feature_name}: {value} (default)")
        
        # Convert to numpy array with shape (1, 1, 12) for LSTM input
        feature_array = np.array(features, dtype=np.float32).reshape(1, 1, -1)
        
        logger.info(f"Feature vector shape: {feature_array.shape}")
        logger.info(f"Feature vector: {feature_array[0][0]}")
        
        return feature_array
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Run prediction using the trained model.
        
        Args:
            features: numpy array of shape (1, 1, 12)
            
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
        label = "Death Event Risk: HIGH" if probability >= 0.5 else "Death Event Risk: LOW"
        
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
