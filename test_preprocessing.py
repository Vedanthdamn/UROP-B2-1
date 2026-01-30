"""
Test suite for preprocessing pipeline

This script validates the HeartFailurePreprocessor class to ensure:
1. Features and target are properly separated
2. Missing values are handled correctly
3. Standardization is consistent and deterministic
4. Serialization/deserialization works correctly
5. Pipeline is reproducible
"""

import numpy as np
import pandas as pd
import os
import tempfile
import sys

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import (
    HeartFailurePreprocessor,
    create_preprocessing_pipeline,
    load_and_preprocess_data
)


def test_feature_target_separation():
    """Test that features and target are properly separated."""
    print("=" * 60)
    print("TEST 1: Feature/Target Separation")
    print("=" * 60)
    
    # Load the heart failure dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X, y = preprocessor.fit_transform(df)
    
    # Validate dimensions
    assert X.shape[0] == df.shape[0], "Number of samples mismatch"
    assert X.shape[1] == df.shape[1] - 1, "Number of features incorrect"
    assert y.shape[0] == df.shape[0], "Number of target values mismatch"
    
    # Validate target values
    assert np.array_equal(y, df['DEATH_EVENT'].values), "Target values don't match"
    
    # Validate feature columns
    expected_features = [col for col in df.columns if col != 'DEATH_EVENT']
    assert preprocessor.get_feature_names() == expected_features, "Feature names mismatch"
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Feature columns: {len(preprocessor.get_feature_names())}")
    print(f"✓ Feature/target separation: PASSED\n")
    return True


def test_missing_value_handling():
    """Test that missing values are handled correctly."""
    print("=" * 60)
    print("TEST 2: Missing Value Handling")
    print("=" * 60)
    
    # Create synthetic data with missing values
    data = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [10.0, np.nan, 30.0, 40.0, 50.0],
        'feature3': [100.0, 200.0, 300.0, 400.0, 500.0],
        'DEATH_EVENT': [0, 1, 0, 1, 0]
    })
    
    # Fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    # Transform data
    X, y = preprocessor.transform(data)
    
    # Validate no missing values in output
    assert not np.isnan(X).any(), "Missing values not properly handled"
    print(f"✓ No NaN values in transformed data")
    
    # Validate median imputation
    expected_median_feature1 = 3.0  # median of [1, 2, 4, 5]
    expected_median_feature2 = 35.0  # median of [10, 30, 40, 50]
    
    assert np.isclose(preprocessor.feature_medians[0], expected_median_feature1), \
        f"Feature1 median incorrect: {preprocessor.feature_medians[0]} != {expected_median_feature1}"
    assert np.isclose(preprocessor.feature_medians[1], expected_median_feature2), \
        f"Feature2 median incorrect: {preprocessor.feature_medians[1]} != {expected_median_feature2}"
    
    print(f"✓ Median imputation working correctly")
    print(f"✓ Missing value handling: PASSED\n")
    return True


def test_standardization_consistency():
    """Test that standardization is consistent and correct."""
    print("=" * 60)
    print("TEST 3: Standardization Consistency")
    print("=" * 60)
    
    # Load real dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X, y = preprocessor.fit_transform(df)
    
    # Validate standardization (mean ≈ 0, std ≈ 1)
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0, ddof=0)
    
    # Check that means are close to 0
    assert np.allclose(feature_means, 0, atol=1e-10), \
        f"Standardized means not close to 0: {feature_means}"
    
    # Check that stds are close to 1
    assert np.allclose(feature_stds, 1, atol=1e-10), \
        f"Standardized stds not close to 1: {feature_stds}"
    
    print(f"✓ Standardized mean range: [{feature_means.min():.2e}, {feature_means.max():.2e}]")
    print(f"✓ Standardized std range: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")
    
    # Test consistency: transform same data twice
    X2, y2 = preprocessor.transform(df)
    assert np.allclose(X, X2), "Transformation not consistent"
    print(f"✓ Transformation is consistent")
    print(f"✓ Standardization: PASSED\n")
    return True


def test_serialization():
    """Test that serialization and deserialization work correctly."""
    print("=" * 60)
    print("TEST 4: Serialization/Deserialization")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X_original, y_original = preprocessor.fit_transform(df)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save preprocessor
        preprocessor.save(tmp_path)
        print(f"✓ Preprocessor saved to {tmp_path}")
        
        # Load preprocessor
        loaded_preprocessor = HeartFailurePreprocessor.load(tmp_path)
        print(f"✓ Preprocessor loaded from {tmp_path}")
        
        # Transform with loaded preprocessor
        X_loaded, y_loaded = loaded_preprocessor.transform(df)
        
        # Validate results are identical
        assert np.allclose(X_original, X_loaded), "Loaded preprocessor produces different results"
        assert np.array_equal(y_original, y_loaded), "Loaded target values don't match"
        
        # Validate attributes are preserved
        assert loaded_preprocessor.is_fitted, "Loaded preprocessor not marked as fitted"
        assert loaded_preprocessor.target_column == preprocessor.target_column, \
            "Target column not preserved"
        assert loaded_preprocessor.get_feature_names() == preprocessor.get_feature_names(), \
            "Feature names not preserved"
        
        print(f"✓ Loaded preprocessor produces identical results")
        print(f"✓ Serialization: PASSED\n")
        return True
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_determinism_reproducibility():
    """Test that preprocessing is deterministic and reproducible."""
    print("=" * 60)
    print("TEST 5: Determinism and Reproducibility")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create two independent preprocessors with same data
    preprocessor1 = create_preprocessing_pipeline()
    X1, y1 = preprocessor1.fit_transform(df)
    
    preprocessor2 = create_preprocessing_pipeline()
    X2, y2 = preprocessor2.fit_transform(df)
    
    # Validate results are identical
    assert np.allclose(X1, X2), "Different preprocessors produce different results"
    assert np.array_equal(y1, y2), "Target values don't match"
    
    # Validate statistics are identical
    assert np.allclose(preprocessor1.feature_means, preprocessor2.feature_means), \
        "Feature means differ"
    assert np.allclose(preprocessor1.feature_stds, preprocessor2.feature_stds), \
        "Feature stds differ"
    assert np.allclose(preprocessor1.feature_medians, preprocessor2.feature_medians), \
        "Feature medians differ"
    
    print(f"✓ Multiple independent preprocessors produce identical results")
    print(f"✓ Statistics are deterministic")
    print(f"✓ Determinism/Reproducibility: PASSED\n")
    return True


def test_inference_mode():
    """Test that preprocessor works without target column (inference mode)."""
    print("=" * 60)
    print("TEST 6: Inference Mode")
    print("=" * 60)
    
    # Load full dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(df)
    
    # Create inference data (without target)
    df_inference = df.drop(columns=['DEATH_EVENT'])
    
    # Transform in inference mode
    X_inference = preprocessor.transform(df_inference, return_target=False)
    
    # Validate shape
    assert X_inference.shape[0] == df_inference.shape[0], "Row count mismatch"
    assert X_inference.shape[1] == df_inference.shape[1], "Column count mismatch"
    
    # Compare with transform on full data
    X_full, _ = preprocessor.transform(df, return_target=True)
    assert np.allclose(X_inference, X_full), "Inference mode produces different results"
    
    print(f"✓ Inference mode (no target) works correctly")
    print(f"✓ Results match transform with target")
    print(f"✓ Inference mode: PASSED\n")
    return True


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("=" * 60)
    print("TEST 7: Error Handling")
    print("=" * 60)
    
    preprocessor = create_preprocessing_pipeline()
    
    # Test transform before fit
    try:
        df = pd.read_csv('data/heart_failure.csv')
        preprocessor.transform(df)
        assert False, "Should raise RuntimeError when transform before fit"
    except RuntimeError as e:
        print(f"✓ Correctly raises RuntimeError when transform before fit")
    
    # Test save before fit
    try:
        preprocessor.save('/tmp/test.pkl')
        assert False, "Should raise RuntimeError when save before fit"
    except RuntimeError:
        print(f"✓ Correctly raises RuntimeError when save before fit")
    
    # Test with missing target column
    preprocessor.fit(pd.read_csv('data/heart_failure.csv'))
    try:
        df_no_target = pd.DataFrame({'feature1': [1, 2, 3]})
        preprocessor.transform(df_no_target, return_target=True)
        assert False, "Should raise ValueError when target column missing"
    except ValueError:
        print(f"✓ Correctly raises ValueError when target column missing")
    
    print(f"✓ Error handling: PASSED\n")
    return True


def test_load_and_preprocess_helper():
    """Test the load_and_preprocess_data helper function."""
    print("=" * 60)
    print("TEST 8: Helper Function (load_and_preprocess_data)")
    print("=" * 60)
    
    # Test loading and preprocessing
    X, y, preprocessor = load_and_preprocess_data('data/heart_failure.csv', fit=True)
    
    # Validate shapes
    assert X.shape[0] == 299, f"Expected 299 rows, got {X.shape[0]}"
    assert X.shape[1] == 12, f"Expected 12 features, got {X.shape[1]}"
    assert y.shape[0] == 299, f"Expected 299 target values, got {y.shape[0]}"
    assert preprocessor.is_fitted, "Preprocessor should be fitted"
    
    print(f"✓ Helper function loads and preprocesses data correctly")
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Preprocessor fitted: {preprocessor.is_fitted}")
    print(f"✓ Helper function: PASSED\n")
    return True


def test_numpy_array_input():
    """Test that preprocessor accepts numpy arrays as input."""
    print("=" * 60)
    print("TEST 9: NumPy Array Input")
    print("=" * 60)
    
    # Load dataset and convert to numpy
    df = pd.read_csv('data/heart_failure.csv')
    data_array = df.values
    
    # Create and fit preprocessor with numpy array
    preprocessor = create_preprocessing_pipeline()
    X, y = preprocessor.fit_transform(data_array)
    
    # Validate results
    assert X.shape[0] == data_array.shape[0], "Row count mismatch"
    assert X.shape[1] == data_array.shape[1] - 1, "Feature count mismatch"
    assert y.shape[0] == data_array.shape[0], "Target count mismatch"
    
    print(f"✓ Preprocessor accepts numpy array input")
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    
    # Test transform with numpy array
    X2, y2 = preprocessor.transform(data_array)
    assert np.allclose(X, X2), "Transform with numpy array produces different results"
    
    print(f"✓ Transform with numpy array works correctly")
    print(f"✓ NumPy array input: PASSED\n")
    return True


def test_constant_features():
    """Test that preprocessor handles constant features correctly."""
    print("=" * 60)
    print("TEST 10: Constant Features Handling")
    print("=" * 60)
    
    # Create data with constant features
    data = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [5.0, 5.0, 5.0, 5.0, 5.0],  # Constant feature
        'feature3': [10.0, 20.0, 30.0, 40.0, 50.0],
        'DEATH_EVENT': [0, 1, 0, 1, 0]
    })
    
    # Fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X, y = preprocessor.fit_transform(data)
    
    # Check that std for constant feature was replaced with 1.0
    assert preprocessor.feature_stds[1] == 1.0, \
        f"Constant feature std should be 1.0, got {preprocessor.feature_stds[1]}"
    
    # Check that no NaN or inf values in output
    assert not np.isnan(X).any(), "NaN values found in output"
    assert not np.isinf(X).any(), "Inf values found in output"
    
    # Check that constant feature has zero variance in output
    assert np.allclose(X[:, 1], 0.0), \
        f"Constant feature should have zero variance after standardization"
    
    print(f"✓ Constant feature std replaced with 1.0")
    print(f"✓ No NaN or Inf values in output")
    print(f"✓ Constant feature properly standardized to 0")
    print(f"✓ Constant features handling: PASSED\n")
    return True


def test_inference_validation():
    """Test that inference mode validates feature columns."""
    print("=" * 60)
    print("TEST 11: Inference Mode Column Validation")
    print("=" * 60)
    
    # Load and fit preprocessor
    df = pd.read_csv('data/heart_failure.csv')
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(df)
    
    # Test with incorrect columns
    try:
        bad_data = pd.DataFrame({
            'wrong_column': [1, 2, 3],
            'another_wrong': [4, 5, 6]
        })
        preprocessor.transform(bad_data, return_target=False)
        assert False, "Should raise ValueError for missing columns"
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for missing columns")
        assert "missing expected feature columns" in str(e).lower(), \
            f"Error message should mention missing columns"
    
    # Test with subset of columns
    try:
        subset_data = df[['age', 'anaemia']].copy()  # Only 2 features
        preprocessor.transform(subset_data, return_target=False)
        assert False, "Should raise ValueError for missing columns"
    except ValueError:
        print(f"✓ Correctly raises ValueError for incomplete feature set")
    
    print(f"✓ Inference mode validation: PASSED\n")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_feature_target_separation,
        test_missing_value_handling,
        test_standardization_consistency,
        test_serialization,
        test_determinism_reproducibility,
        test_inference_mode,
        test_error_handling,
        test_load_and_preprocess_helper,
        test_numpy_array_input,
        test_constant_features,
        test_inference_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
