"""
Test suite for Flower federated client implementation.

This test suite validates the Flower client for federated learning:
1. Client creation and initialization
2. Parameter retrieval (get_parameters)
3. Parameter setting (set_parameters)
4. Local training (fit)
5. Model evaluation (evaluate)
6. Privacy guarantees (no patient data leakage)

Usage:
    python test_federated_client.py
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from federated import FlowerClient, create_flower_client
from models import get_primary_model
from utils.preprocessing import create_preprocessing_pipeline
from utils.client_partitioning import partition_for_federated_clients


def test_client_initialization():
    """Test Flower client initialization."""
    print("Test 1: Client Initialization")
    
    # Create simple test data
    X_train = np.random.randn(50, 1, 12)
    y_train = np.random.randint(0, 2, size=(50,))
    X_val = np.random.randn(10, 1, 12)
    y_val = np.random.randint(0, 2, size=(10,))
    
    # Create model
    model = get_primary_model(input_shape=(1, 12))
    
    # Create client
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs_per_round=1,
        batch_size=16,
        client_id="test_client"
    )
    
    assert client is not None, "Client should not be None"
    assert client.client_id == "test_client", "Client ID mismatch"
    assert client.epochs_per_round == 1, "Epochs per round mismatch"
    assert len(client.X_train) == 50, "Training data size mismatch"
    assert len(client.X_val) == 10, "Validation data size mismatch"
    print("  ✓ Client initialization successful")


def test_get_parameters():
    """Test retrieving model parameters."""
    print("Test 2: Get Parameters")
    
    # Create test data and client
    X_train = np.random.randn(20, 1, 12)
    y_train = np.random.randint(0, 2, size=(20,))
    model = get_primary_model(input_shape=(1, 12))
    
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs_per_round=1,
        client_id="test_client"
    )
    
    # Get parameters
    params = client.get_parameters(config={})
    
    assert params is not None, "Parameters should not be None"
    assert isinstance(params, list), "Parameters should be a list"
    assert len(params) > 0, "Parameters list should not be empty"
    assert all(isinstance(p, np.ndarray) for p in params), "All parameters should be numpy arrays"
    print(f"  ✓ Retrieved {len(params)} parameter arrays")


def test_set_parameters():
    """Test setting model parameters."""
    print("Test 3: Set Parameters")
    
    # Create test data and client
    X_train = np.random.randn(20, 1, 12)
    y_train = np.random.randint(0, 2, size=(20,))
    model = get_primary_model(input_shape=(1, 12))
    
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs_per_round=1,
        client_id="test_client"
    )
    
    # Get initial parameters
    initial_params = client.get_parameters(config={})
    
    # Modify parameters
    modified_params = [p + 0.1 for p in initial_params]
    
    # Set modified parameters
    client.set_parameters(modified_params)
    
    # Verify parameters were set
    new_params = client.get_parameters(config={})
    assert len(new_params) == len(modified_params), "Parameter count mismatch"
    
    # Check that parameters changed
    for i, (new_p, mod_p) in enumerate(zip(new_params, modified_params)):
        assert np.allclose(new_p, mod_p), f"Parameter {i} not set correctly"
    
    print("  ✓ Parameters set successfully")


def test_fit():
    """Test local training (fit)."""
    print("Test 4: Local Training (Fit)")
    
    # Create test data and client
    X_train = np.random.randn(50, 1, 12)
    y_train = np.random.randint(0, 2, size=(50,))
    X_val = np.random.randn(10, 1, 12)
    y_val = np.random.randint(0, 2, size=(10,))
    model = get_primary_model(input_shape=(1, 12))
    
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs_per_round=2,
        batch_size=16,
        client_id="test_client"
    )
    
    # Get initial parameters
    initial_params = client.get_parameters(config={})
    
    # Train
    updated_params, n_samples, metrics = client.fit(initial_params, config={})
    
    # Verify return values
    assert updated_params is not None, "Updated parameters should not be None"
    assert isinstance(updated_params, list), "Updated parameters should be a list"
    assert n_samples == 50, f"Sample count mismatch: expected 50, got {n_samples}"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "train_loss" in metrics, "Metrics should contain train_loss"
    assert "train_accuracy" in metrics, "Metrics should contain train_accuracy"
    assert "val_loss" in metrics, "Metrics should contain val_loss"
    assert "val_accuracy" in metrics, "Metrics should contain val_accuracy"
    
    print(f"  ✓ Training completed with {n_samples} samples")
    print(f"    - Train loss: {metrics['train_loss']:.4f}")
    print(f"    - Train accuracy: {metrics['train_accuracy']:.4f}")


def test_evaluate():
    """Test model evaluation."""
    print("Test 5: Model Evaluation")
    
    # Create test data and client
    X_train = np.random.randn(30, 1, 12)
    y_train = np.random.randint(0, 2, size=(30,))
    X_val = np.random.randn(10, 1, 12)
    y_val = np.random.randint(0, 2, size=(10,))
    model = get_primary_model(input_shape=(1, 12))
    
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs_per_round=1,
        client_id="test_client"
    )
    
    # Get parameters
    params = client.get_parameters(config={})
    
    # Evaluate
    loss, n_samples, metrics = client.evaluate(params, config={})
    
    # Verify return values
    assert loss is not None, "Loss should not be None"
    assert isinstance(loss, float), "Loss should be a float"
    assert n_samples == 10, f"Sample count mismatch: expected 10, got {n_samples}"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "accuracy" in metrics, "Metrics should contain accuracy"
    
    print(f"  ✓ Evaluation completed with {n_samples} samples")
    print(f"    - Loss: {loss:.4f}")
    print(f"    - Accuracy: {metrics['accuracy']:.4f}")


def test_create_flower_client():
    """Test factory function for creating Flower clients."""
    print("Test 6: Factory Function (create_flower_client)")
    
    # Load real data
    data = pd.read_csv('data/heart_failure.csv')
    
    # Partition data
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=3,
        random_seed=42
    )
    
    # Create preprocessor
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    # Create client using factory function
    client = create_flower_client(
        client_data=client_datasets[0],
        preprocessor=preprocessor,
        val_split=0.2,
        epochs_per_round=1,
        batch_size=16,
        client_id="hospital_0"
    )
    
    assert client is not None, "Client should not be None"
    assert isinstance(client, FlowerClient), "Client should be FlowerClient instance"
    assert client.client_id == "hospital_0", "Client ID mismatch"
    assert client.X_train.shape[1] == 1, "Sequence length should be 1"
    assert client.X_train.shape[2] == 12, "Feature count should be 12"
    
    print(f"  ✓ Client created successfully")
    print(f"    - Train samples: {len(client.X_train)}")
    print(f"    - Val samples: {len(client.X_val) if client.X_val is not None else 0}")


def test_no_patient_data_in_parameters():
    """Test that parameters contain only model weights, not patient data."""
    print("Test 7: Privacy - No Patient Data in Parameters")
    
    # Create test data with known values
    X_train = np.ones((20, 1, 12)) * 999.0  # Use distinctive values
    y_train = np.ones(20)  # All ones
    model = get_primary_model(input_shape=(1, 12))
    
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs_per_round=1,
        client_id="test_client"
    )
    
    # Get parameters
    params = client.get_parameters(config={})
    
    # Verify that parameters do not contain patient data
    # Patient data has value 999.0, model weights should be different
    for i, param in enumerate(params):
        # Check that no values are 999.0 (patient data marker)
        has_patient_data = np.any(np.abs(param - 999.0) < 1e-6)
        assert not has_patient_data, f"Parameter {i} appears to contain patient data"
    
    print("  ✓ Parameters contain only model weights (no patient data)")


def test_multiple_rounds():
    """Test multiple training rounds."""
    print("Test 8: Multiple Training Rounds")
    
    # Create test data and client
    X_train = np.random.randn(40, 1, 12)
    y_train = np.random.randint(0, 2, size=(40,))
    model = get_primary_model(input_shape=(1, 12))
    
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs_per_round=1,
        client_id="test_client"
    )
    
    # Simulate 3 rounds of training
    params = client.get_parameters(config={})
    
    for round_num in range(3):
        updated_params, n_samples, metrics = client.fit(params, config={})
        params = updated_params
        
        assert len(params) > 0, f"Round {round_num}: No parameters returned"
        assert n_samples == 40, f"Round {round_num}: Sample count mismatch"
        assert "train_loss" in metrics, f"Round {round_num}: Missing train_loss"
    
    print("  ✓ Multiple rounds completed successfully")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 80)
    print("FEDERATED CLIENT TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        test_client_initialization,
        test_get_parameters,
        test_set_parameters,
        test_fit,
        test_evaluate,
        test_create_flower_client,
        test_no_patient_data_in_parameters,
        test_multiple_rounds,
    ]
    
    failed_tests = []
    
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed_tests.append((test_func.__name__, str(e)))
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed_tests.append((test_func.__name__, str(e)))
        print()
    
    print("=" * 80)
    if not failed_tests:
        print("ALL TESTS PASSED ✓")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"FAILED: {len(failed_tests)} test(s)")
        print("=" * 80)
        for test_name, error in failed_tests:
            print(f"\n{test_name}:")
            print(f"  {error}")
        print("\n" + "=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
