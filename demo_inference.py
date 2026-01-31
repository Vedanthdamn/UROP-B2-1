"""
End-to-End Demo: Upload → Prediction

This script demonstrates the complete inference pipeline:
1. Create sample patient data
2. Save as CSV
3. Load inference pipeline
4. Make predictions
5. Display results

Usage:
    python demo_inference.py

Author: Federated Learning Medical AI Project
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from inference import InferencePipeline


def create_sample_patient_data() -> pd.DataFrame:
    """
    Create sample patient data for demonstration.
    
    Returns:
        DataFrame with sample patient records
    """
    # Sample patients with different profiles
    patients = [
        # High-risk patient (older, high creatinine, low ejection fraction)
        {
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
        },
        # Low-risk patient (younger, normal values, high ejection fraction)
        {
            'age': 45,
            'anaemia': 0,
            'creatinine_phosphokinase': 120,
            'diabetes': 0,
            'ejection_fraction': 60,
            'high_blood_pressure': 0,
            'platelets': 300000,
            'serum_creatinine': 1.0,
            'serum_sodium': 140,
            'sex': 0,
            'smoking': 0,
            'time': 100
        },
        # Medium-risk patient
        {
            'age': 60,
            'anaemia': 0,
            'creatinine_phosphokinase': 300,
            'diabetes': 1,
            'ejection_fraction': 40,
            'high_blood_pressure': 1,
            'platelets': 250000,
            'serum_creatinine': 1.4,
            'serum_sodium': 135,
            'sex': 1,
            'smoking': 1,
            'time': 50
        }
    ]
    
    return pd.DataFrame(patients)


def main():
    """Run the end-to-end demo."""
    print("=" * 80)
    print("INFERENCE PIPELINE DEMO")
    print("End-to-End: Upload → Prediction")
    print("=" * 80)
    print()
    
    # Step 1: Create sample data
    print("Step 1: Creating sample patient data...")
    sample_data = create_sample_patient_data()
    print(f"Created {len(sample_data)} sample patients")
    print()
    
    # Step 2: Save to CSV
    csv_path = '/tmp/sample_patients.csv'
    print(f"Step 2: Saving to CSV: {csv_path}")
    sample_data.to_csv(csv_path, index=False)
    print("✓ Data saved")
    print()
    
    # Step 3: Initialize inference pipeline
    print("Step 3: Initializing inference pipeline...")
    pipeline = InferencePipeline()
    
    # Check if saved weights exist
    weights_path = 'logs/model_weights.h5'
    
    try:
        if os.path.exists(weights_path):
            print(f"  - Loading model from saved weights: {weights_path}")
            pipeline.load_model_from_history(
                history_path='logs/training_history.json',
                data_path='data/heart_failure.csv',
                model_weights_path=weights_path
            )
        else:
            print("  - No saved weights found.")
            print("  - Creating a fresh model for demonstration purposes...")
            print("  - NOTE: This model is not trained. For real predictions, run:")
            print("    python save_model_weights.py")
            
            # Load just the preprocessor and create an untrained model
            from models import get_primary_model
            from utils.preprocessing import create_preprocessing_pipeline
            import pandas as pd
            
            data = pd.read_csv('data/heart_failure.csv')
            pipeline.preprocessor = create_preprocessing_pipeline()
            pipeline.preprocessor.fit(data)
            pipeline.model = get_primary_model(input_shape=(1, 12))
            pipeline.is_loaded = True
            
            print("  - Using fresh (untrained) model for demonstration")
            
        print("✓ Pipeline ready")
    except Exception as e:
        print(f"✗ Error initializing pipeline: {e}")
        return 1
    
    print()
    
    # Step 4: Make predictions
    print("Step 4: Making predictions...")
    print("-" * 80)
    
    for idx in range(len(sample_data)):
        patient = sample_data.iloc[idx:idx+1]
        
        print(f"\nPatient {idx + 1}:")
        print(f"  Age: {int(patient['age'].values[0])}")
        print(f"  Ejection Fraction: {int(patient['ejection_fraction'].values[0])}%")
        print(f"  Serum Creatinine: {float(patient['serum_creatinine'].values[0])}")
        print(f"  Serum Sodium: {int(patient['serum_sodium'].values[0])}")
        
        try:
            prediction, confidence = pipeline.predict(patient)
            
            print(f"\n  → PREDICTION: Death Event = {prediction}")
            print(f"  → CONFIDENCE: {confidence * 100:.2f}%")
            
            if prediction == "Yes":
                print(f"  → RISK LEVEL: HIGH RISK")
            else:
                print(f"  → RISK LEVEL: LOW RISK")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print()
    print("-" * 80)
    print()
    
    # Step 5: Batch prediction from CSV
    print("Step 5: Batch prediction from CSV file...")
    patients_df = pd.read_csv(csv_path)
    
    results = []
    for idx in range(len(patients_df)):
        patient_row = patients_df.iloc[idx:idx+1]
        prediction, confidence = pipeline.predict(patient_row)
        results.append({
            'Patient ID': idx + 1,
            'Prediction': prediction,
            'Confidence': f"{confidence * 100:.2f}%",
            'Risk Level': 'HIGH' if prediction == 'Yes' else 'LOW'
        })
    
    results_df = pd.DataFrame(results)
    print("\nBatch Prediction Results:")
    print(results_df.to_string(index=False))
    print()
    
    # Important notes
    print("=" * 80)
    print("IMPORTANT NOTES:")
    print("=" * 80)
    print("✓ NO training occurred during inference")
    print("✓ NO patient data was stored")
    print("✓ All processing done in-memory only")
    print("✓ Model loaded from trained artifacts")
    print()
    print("⚠️  DISCLAIMER:")
    print("This system is for research purposes only and is NOT a medical diagnosis.")
    print("Always consult qualified healthcare professionals for medical advice.")
    print("=" * 80)
    print()
    
    # Clean up
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"✓ Cleaned up temporary file: {csv_path}")
    
    print("\nDemo completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
