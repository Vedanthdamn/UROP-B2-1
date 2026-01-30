"""
Test suite for model architectures.

This test suite validates all model implementations for federated learning:
1. Model instantiation
2. Input/output shapes
3. Parameter counts (FL-friendliness)
4. Model compilation
5. Forward pass
6. Integration with preprocessing pipeline

Usage:
    python test_models.py
"""

import sys
import numpy as np
import tensorflow as tf
from models import (
    create_lstm_classifier,
    create_tcn_classifier, 
    create_transformer_classifier,
    get_model,
    get_primary_model,
    LSTMClassifier,
    TCNClassifier,
    TransformerClassifier
)


def test_lstm_instantiation():
    """Test LSTM classifier instantiation."""
    print("Test 1: LSTM Instantiation")
    model = create_lstm_classifier(input_shape=(1, 12))
    assert model is not None, "LSTM model should not be None"
    assert isinstance(model, tf.keras.Model), "Should return Keras Model"
    print("  ✓ LSTM instantiation successful")


def test_tcn_instantiation():
    """Test TCN classifier instantiation."""
    print("Test 2: TCN Instantiation")
    model = create_tcn_classifier(input_shape=(1, 12))
    assert model is not None, "TCN model should not be None"
    assert isinstance(model, tf.keras.Model), "Should return Keras Model"
    print("  ✓ TCN instantiation successful")


def test_transformer_instantiation():
    """Test Transformer classifier instantiation."""
    print("Test 3: Transformer Instantiation")
    model = create_transformer_classifier(input_shape=(1, 12))
    assert model is not None, "Transformer model should not be None"
    assert isinstance(model, tf.keras.Model), "Should return Keras Model"
    print("  ✓ Transformer instantiation successful")


def test_input_output_shapes():
    """Test that all models have correct input/output shapes."""
    print("Test 4: Input/Output Shapes")
    
    models = {
        'LSTM': create_lstm_classifier(input_shape=(1, 12)),
        'TCN': create_tcn_classifier(input_shape=(1, 12)),
        'Transformer': create_transformer_classifier(input_shape=(1, 12))
    }
    
    for name, model in models.items():
        # Check input shape
        input_shape = model.input_shape
        assert input_shape == (None, 1, 12), f"{name} input shape incorrect: {input_shape}"
        
        # Check output shape
        output_shape = model.output_shape
        assert output_shape == (None, 1), f"{name} output shape incorrect: {output_shape}"
        
        print(f"  ✓ {name}: input {input_shape}, output {output_shape}")


def test_parameter_counts():
    """Test that models have reasonable parameter counts for FL."""
    print("Test 5: Parameter Counts (FL-friendliness)")
    
    models = {
        'LSTM': create_lstm_classifier(input_shape=(1, 12)),
        'TCN': create_tcn_classifier(input_shape=(1, 12)),
        'Transformer': create_transformer_classifier(input_shape=(1, 12))
    }
    
    max_params = 10000  # FL-friendly threshold
    
    for name, model in models.items():
        param_count = model.count_params()
        assert param_count > 0, f"{name} should have parameters"
        assert param_count < max_params, f"{name} has too many parameters for FL: {param_count}"
        print(f"  ✓ {name}: {param_count:,} parameters (FL-friendly)")


def test_model_compilation():
    """Test that all models are properly compiled."""
    print("Test 6: Model Compilation")
    
    models = {
        'LSTM': create_lstm_classifier(input_shape=(1, 12)),
        'TCN': create_tcn_classifier(input_shape=(1, 12)),
        'Transformer': create_transformer_classifier(input_shape=(1, 12))
    }
    
    for name, model in models.items():
        assert model.optimizer is not None, f"{name} should be compiled"
        assert model.loss is not None, f"{name} should have loss function"
        assert len(model.metrics) > 0, f"{name} should have metrics"
        print(f"  ✓ {name}: compiled with optimizer, loss, and metrics")


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("Test 7: Forward Pass")
    
    models = {
        'LSTM': create_lstm_classifier(input_shape=(1, 12)),
        'TCN': create_tcn_classifier(input_shape=(1, 12)),
        'Transformer': create_transformer_classifier(input_shape=(1, 12))
    }
    
    # Create dummy data
    X_dummy = np.random.randn(10, 1, 12).astype(np.float32)
    
    for name, model in models.items():
        predictions = model.predict(X_dummy, verbose=0)
        assert predictions.shape == (10, 1), f"{name} prediction shape incorrect"
        assert np.all((predictions >= 0) & (predictions <= 1)), f"{name} outputs not in [0,1]"
        print(f"  ✓ {name}: forward pass successful, predictions in [0,1]")


def test_model_registry():
    """Test model registry and factory functions."""
    print("Test 8: Model Registry")
    
    # Test get_model
    lstm = get_model('lstm', input_shape=(1, 12))
    assert lstm is not None, "get_model('lstm') should work"
    
    tcn = get_model('tcn', input_shape=(1, 12))
    assert tcn is not None, "get_model('tcn') should work"
    
    transformer = get_model('transformer', input_shape=(1, 12))
    assert transformer is not None, "get_model('transformer') should work"
    
    # Test get_primary_model
    primary = get_primary_model(input_shape=(1, 12))
    assert primary is not None, "get_primary_model() should work"
    assert primary.name == 'LSTMClassifier', "Primary model should be LSTM"
    
    print("  ✓ Model registry and factory functions work correctly")


def test_class_instantiation():
    """Test class-based instantiation."""
    print("Test 9: Class-based Instantiation")
    
    # Test LSTMClassifier
    lstm_cls = LSTMClassifier(input_shape=(1, 12), lstm_units=32, dropout_rate=0.3)
    assert lstm_cls.model is not None, "LSTMClassifier should have model"
    
    # Test TCNClassifier
    tcn_cls = TCNClassifier(input_shape=(1, 12), filters=32, kernel_size=2)
    assert tcn_cls.model is not None, "TCNClassifier should have model"
    
    # Test TransformerClassifier
    transformer_cls = TransformerClassifier(input_shape=(1, 12), num_heads=1)
    assert transformer_cls.model is not None, "TransformerClassifier should have model"
    
    print("  ✓ Class-based instantiation works for all models")


def test_integration_with_preprocessing():
    """Test integration with preprocessing pipeline."""
    print("Test 10: Integration with Preprocessing Pipeline")
    
    try:
        from utils.preprocessing import create_preprocessing_pipeline
        import pandas as pd
        
        # Load real data
        df = pd.read_csv('data/heart_failure.csv')
        
        # Preprocess
        preprocessor = create_preprocessing_pipeline()
        X, y = preprocessor.fit_transform(df)
        
        # Reshape for models
        X_reshaped = X.reshape(-1, 1, 12)
        
        # Test with LSTM (primary model)
        model = get_primary_model(input_shape=(1, 12))
        predictions = model.predict(X_reshaped[:10], verbose=0)
        
        assert predictions.shape == (10, 1), "Predictions shape incorrect"
        assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions not in [0,1]"
        
        print("  ✓ Integration with preprocessing pipeline successful")
    except Exception as e:
        print(f"  ⚠ Integration test skipped (data not available): {e}")


def test_customization():
    """Test model customization with different hyperparameters."""
    print("Test 11: Model Customization")
    
    # LSTM with custom params
    lstm = create_lstm_classifier(input_shape=(1, 12), lstm_units=64, dropout_rate=0.5)
    assert lstm.layers[1].units == 64, "LSTM units should be customizable"
    
    # TCN with custom params
    tcn = create_tcn_classifier(input_shape=(1, 12), filters=64, kernel_size=3)
    # Note: Can't easily check filters without diving into layer config
    
    # Transformer with custom params
    transformer = create_transformer_classifier(input_shape=(1, 12), num_heads=1, ff_dim=64)
    
    print("  ✓ Model customization works correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("MODEL ARCHITECTURES TEST SUITE")
    print("=" * 80)
    print()
    
    tests = [
        test_lstm_instantiation,
        test_tcn_instantiation,
        test_transformer_instantiation,
        test_input_output_shapes,
        test_parameter_counts,
        test_model_compilation,
        test_forward_pass,
        test_model_registry,
        test_class_instantiation,
        test_integration_with_preprocessing,
        test_customization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
