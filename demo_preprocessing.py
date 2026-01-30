"""
Demonstration script for preprocessing pipeline

This script demonstrates how to use the HeartFailurePreprocessor
for federated learning scenarios.

Usage:
    python demo_preprocessing.py
"""

import numpy as np
import pandas as pd
from utils.preprocessing import (
    create_preprocessing_pipeline,
    load_and_preprocess_data
)


def main():
    print("=" * 80)
    print("PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 80)
    print()
    
    # ===== METHOD 1: Using the convenience function =====
    print("-" * 80)
    print("METHOD 1: Using load_and_preprocess_data (Recommended for most cases)")
    print("-" * 80)
    
    X, y, preprocessor = load_and_preprocess_data('data/heart_failure.csv', fit=True)
    
    print(f"Loaded and preprocessed data:")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Target shape: {y.shape}")
    print(f"  - Feature names: {preprocessor.get_feature_names()}")
    print()
    
    # ===== METHOD 2: Manual preprocessing =====
    print("-" * 80)
    print("METHOD 2: Manual preprocessing (More control)")
    print("-" * 80)
    
    # Load data
    df = pd.read_csv('data/heart_failure.csv')
    print(f"Loaded raw data: {df.shape}")
    
    # Create preprocessor
    preprocessor = create_preprocessing_pipeline()
    print(f"Created preprocessor: {preprocessor}")
    
    # Fit and transform
    X_train, y_train = preprocessor.fit_transform(df)
    print(f"Fitted and transformed:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print()
    
    # ===== Save preprocessor for federated clients =====
    print("-" * 80)
    print("SAVING PREPROCESSOR FOR FEDERATED CLIENTS")
    print("-" * 80)
    
    save_path = '/tmp/heart_failure_preprocessor.pkl'
    preprocessor.save(save_path)
    print(f"✓ Preprocessor saved to: {save_path}")
    print(f"  This file can be distributed to all federated clients")
    print()
    
    # ===== Load and use preprocessor (as federated client would) =====
    print("-" * 80)
    print("LOADING PREPROCESSOR (Federated Client Simulation)")
    print("-" * 80)
    
    from utils.preprocessing import HeartFailurePreprocessor
    
    # Load the saved preprocessor
    client_preprocessor = HeartFailurePreprocessor.load(save_path)
    print(f"✓ Loaded preprocessor: {client_preprocessor}")
    
    # Use it to transform new data
    X_test, y_test = client_preprocessor.transform(df)
    print(f"Transformed data using loaded preprocessor:")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_test shape: {y_test.shape}")
    
    # Verify consistency
    assert np.allclose(X_train, X_test), "Results should be identical"
    print(f"✓ Results are identical to original preprocessing")
    print()
    
    # ===== Show preprocessing statistics =====
    print("-" * 80)
    print("PREPROCESSING STATISTICS")
    print("-" * 80)
    
    print(f"Feature statistics computed from training data:")
    print(f"  - Feature means (first 5): {preprocessor.feature_means[:5]}")
    print(f"  - Feature stds (first 5): {preprocessor.feature_stds[:5]}")
    print(f"  - Feature medians (first 5): {preprocessor.feature_medians[:5]}")
    print()
    
    # ===== Verify standardization =====
    print("-" * 80)
    print("STANDARDIZATION VERIFICATION")
    print("-" * 80)
    
    # Check standardization properties
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0, ddof=0)
    
    print(f"After standardization:")
    print(f"  - Mean (should be ~0): min={X_mean.min():.2e}, max={X_mean.max():.2e}")
    print(f"  - Std (should be ~1): min={X_std.min():.4f}, max={X_std.max():.4f}")
    print()
    
    # ===== Inference mode demonstration =====
    print("-" * 80)
    print("INFERENCE MODE (Without Target)")
    print("-" * 80)
    
    # Simulate inference data (without DEATH_EVENT column)
    df_inference = df.drop(columns=['DEATH_EVENT'])
    X_inference = client_preprocessor.transform(df_inference, return_target=False)
    
    print(f"Inference data preprocessed:")
    print(f"  - X_inference shape: {X_inference.shape}")
    print(f"✓ Inference mode works without target column")
    print()
    
    # ===== Summary =====
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Key Points:")
    print("  1. ✓ Features and target are properly separated")
    print("  2. ✓ Missing values handled with median imputation")
    print("  3. ✓ Standardization ensures mean≈0 and std≈1")
    print("  4. ✓ Preprocessor is serializable for federated deployment")
    print("  5. ✓ Results are deterministic and reproducible")
    print("  6. ✓ Same pipeline can be used by all federated clients")
    print()
    print("For federated learning:")
    print("  - Fit preprocessor on combined/representative training data")
    print("  - Save preprocessor to disk (.pkl file)")
    print("  - Distribute the .pkl file to all federated clients")
    print("  - Each client loads and uses the same preprocessor")
    print("  - This ensures consistency across all clients")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
