"""
Test suite for Differential Privacy in Federated Learning.

This test suite validates the DP implementation:
1. DPConfig creation and validation
2. Gradient clipping
3. Gaussian noise addition
4. DP application to model updates
5. Integration with FlowerClient
6. Privacy budget tracking
7. Loss logging after DP

Usage:
    python test_differential_privacy.py
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from federated import (
    FlowerClient,
    create_flower_client,
    DifferentialPrivacy,
    DPConfig,
    create_dp_config
)
from models import get_primary_model
from utils.preprocessing import create_preprocessing_pipeline
from utils.client_partitioning import partition_for_federated_clients


def test_dp_config_creation():
    """Test DPConfig creation and validation."""
    print("Test 1: DPConfig Creation and Validation")
    
    # Test valid config
    config = create_dp_config(
        epsilon=1.0,
        delta=1e-5,
        l2_norm_clip=1.0,
        enabled=True
    )
    
    assert config.epsilon == 1.0, "Epsilon mismatch"
    assert config.delta == 1e-5, "Delta mismatch"
    assert config.l2_norm_clip == 1.0, "L2 norm clip mismatch"
    assert config.enabled is True, "Enabled flag mismatch"
    assert config.noise_multiplier is not None, "Noise multiplier should be computed"
    assert config.noise_multiplier > 0, "Noise multiplier should be positive"
    
    print(f"  ✓ Valid DPConfig created: epsilon={config.epsilon}, delta={config.delta}")
    print(f"    Computed noise_multiplier={config.noise_multiplier:.4f}")
    
    # Test invalid epsilon
    try:
        invalid_config = DPConfig(epsilon=-1.0, delta=1e-5, l2_norm_clip=1.0, enabled=True)
        assert False, "Should raise ValueError for negative epsilon"
    except ValueError as e:
        print(f"  ✓ Correctly rejected negative epsilon: {e}")
    
    # Test invalid delta
    try:
        invalid_config = DPConfig(epsilon=1.0, delta=1.5, l2_norm_clip=1.0, enabled=True)
        assert False, "Should raise ValueError for delta >= 1"
    except ValueError as e:
        print(f"  ✓ Correctly rejected invalid delta: {e}")
    
    # Test disabled DP
    disabled_config = create_dp_config(enabled=False)
    assert disabled_config.enabled is False, "DP should be disabled"
    print("  ✓ Disabled DPConfig created successfully")


def test_gradient_clipping():
    """Test gradient clipping mechanism."""
    print("\nTest 2: Gradient Clipping")
    
    # Create DP mechanism
    config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
    dp = DifferentialPrivacy(config)
    
    # Create test gradients with large norm
    gradients = [
        np.array([[2.0, 3.0], [4.0, 5.0]]),  # L2 norm contribution: sqrt(4+9+16+25) = sqrt(54)
        np.array([1.0, 2.0])  # L2 norm contribution: sqrt(1+4) = sqrt(5)
    ]
    
    # Compute expected global norm
    global_norm = np.sqrt(54 + 5)  # sqrt(59) ≈ 7.68
    assert global_norm > config.l2_norm_clip, "Test gradients should exceed clip threshold"
    
    # Apply clipping
    clipped_gradients = dp.clip_gradients(gradients)
    
    # Verify clipping occurred
    clipped_norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped_gradients))
    assert abs(clipped_norm - config.l2_norm_clip) < 1e-5, \
        f"Clipped norm {clipped_norm} should equal clip threshold {config.l2_norm_clip}"
    
    print(f"  ✓ Gradients clipped: original_norm={global_norm:.4f}, clipped_norm={clipped_norm:.4f}")
    
    # Test no clipping needed (small gradients)
    small_gradients = [np.array([0.1, 0.1]), np.array([0.1])]
    unclipped = dp.clip_gradients(small_gradients)
    small_norm = np.sqrt(sum(np.sum(g ** 2) for g in small_gradients))
    assert small_norm < config.l2_norm_clip, "Small gradients should not need clipping"
    
    for orig, unclip in zip(small_gradients, unclipped):
        assert np.allclose(orig, unclip), "Small gradients should not be modified"
    
    print(f"  ✓ No clipping for small gradients: norm={small_norm:.4f} < clip={config.l2_norm_clip}")


def test_noise_addition():
    """Test Gaussian noise addition."""
    print("\nTest 3: Gaussian Noise Addition")
    
    # Create DP mechanism
    config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
    dp = DifferentialPrivacy(config)
    
    # Create test gradients
    gradients = [
        np.ones((10, 10)),
        np.ones((5,))
    ]
    
    # Add noise
    np.random.seed(42)  # For reproducibility
    noisy_gradients = dp.add_noise(gradients)
    
    # Verify noise was added
    assert len(noisy_gradients) == len(gradients), "Should return same number of arrays"
    for i, (orig, noisy) in enumerate(zip(gradients, noisy_gradients)):
        assert orig.shape == noisy.shape, f"Shape mismatch in gradient {i}"
        assert not np.allclose(orig, noisy), f"Gradient {i} should be different after noise addition"
    
    # Verify noise statistics (approximate, since it's random)
    expected_stddev = config.l2_norm_clip * config.noise_multiplier
    noise_samples = []
    for orig, noisy in zip(gradients, noisy_gradients):
        noise = noisy - orig
        noise_samples.extend(noise.flatten())
    
    actual_stddev = np.std(noise_samples)
    # Allow 50% margin due to randomness and small sample
    assert abs(actual_stddev - expected_stddev) / expected_stddev < 0.5, \
        f"Noise stddev {actual_stddev:.4f} should be close to expected {expected_stddev:.4f}"
    
    print(f"  ✓ Noise added successfully: expected_stddev={expected_stddev:.4f}, actual={actual_stddev:.4f}")


def test_dp_to_updates():
    """Test DP application to model weight updates."""
    print("\nTest 4: DP Application to Model Updates")
    
    # Create DP mechanism
    config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
    dp = DifferentialPrivacy(config)
    
    # Create test weights (before and after training)
    original_weights = [
        np.ones((5, 5)),
        np.ones((5,))
    ]
    
    updated_weights = [
        np.ones((5, 5)) + 0.5,  # All weights increased by 0.5
        np.ones((5,)) + 0.5
    ]
    
    # Apply DP
    np.random.seed(42)
    dp_weights, metrics = dp.apply_dp_to_updates(original_weights, updated_weights)
    
    # Verify return values
    assert len(dp_weights) == len(original_weights), "Should return same number of weight arrays"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert metrics["dp_enabled"] is True, "DP should be enabled"
    assert "dp_epsilon" in metrics, "Should include epsilon"
    assert "dp_delta" in metrics, "Should include delta"
    assert "dp_l2_norm_clip" in metrics, "Should include l2_norm_clip"
    assert "dp_noise_multiplier" in metrics, "Should include noise_multiplier"
    assert "dp_original_update_norm" in metrics, "Should include original update norm"
    assert "dp_clipped_update_norm" in metrics, "Should include clipped update norm"
    assert "dp_noisy_update_norm" in metrics, "Should include noisy update norm"
    
    # Verify DP was applied
    for i, (orig, dp_w) in enumerate(zip(updated_weights, dp_weights)):
        assert not np.allclose(orig, dp_w), f"Weight {i} should be different after DP"
    
    print(f"  ✓ DP applied to updates")
    print(f"    - epsilon: {metrics['dp_epsilon']}")
    print(f"    - delta: {metrics['dp_delta']}")
    print(f"    - l2_norm_clip: {metrics['dp_l2_norm_clip']}")
    print(f"    - original_update_norm: {metrics['dp_original_update_norm']:.4f}")
    print(f"    - clipped_update_norm: {metrics['dp_clipped_update_norm']:.4f}")


def test_disabled_dp():
    """Test that DP can be disabled."""
    print("\nTest 5: Disabled DP")
    
    # Create disabled DP mechanism
    config = create_dp_config(enabled=False)
    dp = DifferentialPrivacy(config)
    
    # Create test weights
    original_weights = [np.ones((5, 5))]
    updated_weights = [np.ones((5, 5)) + 0.5]
    
    # Apply "DP" (should do nothing)
    dp_weights, metrics = dp.apply_dp_to_updates(original_weights, updated_weights)
    
    # Verify no DP was applied
    assert metrics["dp_enabled"] is False, "DP should be disabled"
    for orig, dp_w in zip(updated_weights, dp_weights):
        assert np.allclose(orig, dp_w), "Weights should be unchanged when DP is disabled"
    
    print("  ✓ Disabled DP correctly passes through weights unchanged")


def test_flower_client_with_dp():
    """Test FlowerClient with DP integration."""
    print("\nTest 6: FlowerClient with DP")
    
    # Create test data
    X_train = np.random.randn(50, 1, 12)
    y_train = np.random.randint(0, 2, size=(50,))
    X_val = np.random.randn(10, 1, 12)
    y_val = np.random.randint(0, 2, size=(10,))
    
    # Create DP config
    dp_config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
    
    # Create model and client
    model = get_primary_model(input_shape=(1, 12))
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs_per_round=1,
        batch_size=16,
        client_id="test_dp_client",
        dp_config=dp_config
    )
    
    # Verify DP mechanism is initialized
    assert client.dp_mechanism is not None, "DP mechanism should be initialized"
    assert client.dp_config is not None, "DP config should be set"
    
    # Get initial parameters
    initial_params = client.get_parameters(config={})
    
    # Train with DP
    np.random.seed(42)
    updated_params, n_samples, metrics = client.fit(initial_params, config={})
    
    # Verify DP metrics are included
    assert "dp_enabled" in metrics, "Should include dp_enabled"
    assert metrics["dp_enabled"] is True, "DP should be enabled"
    assert "dp_epsilon" in metrics, "Should include dp_epsilon"
    assert "dp_delta" in metrics, "Should include dp_delta"
    assert "dp_l2_norm_clip" in metrics, "Should include dp_l2_norm_clip"
    assert "train_loss_after_dp" in metrics, "Should include train_loss_after_dp"
    
    # Verify training completed
    assert n_samples == 50, "Sample count should be correct"
    assert "train_loss" in metrics, "Should include train_loss"
    assert "train_accuracy" in metrics, "Should include train_accuracy"
    
    print(f"  ✓ FlowerClient with DP trained successfully")
    print(f"    - DP epsilon: {metrics['dp_epsilon']}")
    print(f"    - DP delta: {metrics['dp_delta']}")
    print(f"    - Train loss (before DP): {metrics['train_loss']:.4f}")
    print(f"    - Train loss (after DP): {metrics['train_loss_after_dp']:.4f}")


def test_flower_client_without_dp():
    """Test FlowerClient without DP."""
    print("\nTest 7: FlowerClient without DP")
    
    # Create test data
    X_train = np.random.randn(50, 1, 12)
    y_train = np.random.randint(0, 2, size=(50,))
    
    # Create model and client (no DP config)
    model = get_primary_model(input_shape=(1, 12))
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs_per_round=1,
        batch_size=16,
        client_id="test_no_dp_client"
    )
    
    # Verify DP mechanism is not initialized
    assert client.dp_mechanism is None, "DP mechanism should not be initialized"
    
    # Get initial parameters
    initial_params = client.get_parameters(config={})
    
    # Train without DP
    updated_params, n_samples, metrics = client.fit(initial_params, config={})
    
    # Verify no DP metrics
    assert "dp_enabled" in metrics, "Should include dp_enabled"
    assert metrics["dp_enabled"] is False, "DP should be disabled"
    assert "train_loss_after_dp" not in metrics, "Should not include train_loss_after_dp"
    
    print("  ✓ FlowerClient without DP trained successfully")
    print(f"    - DP enabled: {metrics['dp_enabled']}")


def test_factory_function_with_dp():
    """Test create_flower_client factory function with DP."""
    print("\nTest 8: Factory Function with DP")
    
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
    
    # Create DP config
    dp_config = create_dp_config(epsilon=2.0, delta=1e-5, l2_norm_clip=1.5)
    
    # Create client using factory function with DP
    client = create_flower_client(
        client_data=client_datasets[0],
        preprocessor=preprocessor,
        val_split=0.2,
        epochs_per_round=1,
        batch_size=16,
        client_id="hospital_0_dp",
        dp_config=dp_config
    )
    
    # Verify client was created with DP
    assert client is not None, "Client should not be None"
    assert isinstance(client, FlowerClient), "Client should be FlowerClient instance"
    assert client.dp_mechanism is not None, "DP mechanism should be initialized"
    assert client.dp_config.epsilon == 2.0, "Epsilon should match"
    assert client.dp_config.l2_norm_clip == 1.5, "L2 norm clip should match"
    
    print(f"  ✓ Client created with DP via factory function")
    print(f"    - Train samples: {len(client.X_train)}")
    print(f"    - DP epsilon: {client.dp_config.epsilon}")
    print(f"    - DP l2_norm_clip: {client.dp_config.l2_norm_clip}")


def test_privacy_budget_logging():
    """Test that privacy budget parameters are properly logged."""
    print("\nTest 9: Privacy Budget Logging")
    
    # Create test data and client with DP
    X_train = np.random.randn(30, 1, 12)
    y_train = np.random.randint(0, 2, size=(30,))
    
    dp_config = create_dp_config(
        epsilon=0.5,
        delta=1e-6,
        l2_norm_clip=2.0,
        enabled=True
    )
    
    model = get_primary_model(input_shape=(1, 12))
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs_per_round=1,
        batch_size=16,
        client_id="test_logging_client",
        dp_config=dp_config
    )
    
    # Get initial parameters and train
    initial_params = client.get_parameters(config={})
    np.random.seed(42)
    _, _, metrics = client.fit(initial_params, config={})
    
    # Verify all required privacy budget parameters are logged
    assert metrics["dp_epsilon"] == 0.5, "Epsilon should be logged"
    assert metrics["dp_delta"] == 1e-6, "Delta should be logged"
    assert metrics["dp_l2_norm_clip"] == 2.0, "L2 norm clip should be logged"
    assert "dp_noise_multiplier" in metrics, "Noise multiplier should be logged"
    assert metrics["dp_noise_multiplier"] > 0, "Noise multiplier should be positive"
    
    print("  ✓ All privacy budget parameters logged correctly")
    print(f"    - epsilon: {metrics['dp_epsilon']}")
    print(f"    - delta: {metrics['dp_delta']}")
    print(f"    - l2_norm_clip: {metrics['dp_l2_norm_clip']}")
    print(f"    - noise_multiplier: {metrics['dp_noise_multiplier']:.4f}")


def test_loss_after_dp():
    """Test that local training loss is logged after DP application."""
    print("\nTest 10: Loss After DP Application")
    
    # Create test data and client with DP
    X_train = np.random.randn(40, 1, 12)
    y_train = np.random.randint(0, 2, size=(40,))
    
    dp_config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
    
    model = get_primary_model(input_shape=(1, 12))
    client = FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs_per_round=1,
        batch_size=16,
        client_id="test_loss_client",
        dp_config=dp_config
    )
    
    # Train
    initial_params = client.get_parameters(config={})
    np.random.seed(42)
    _, _, metrics = client.fit(initial_params, config={})
    
    # Verify loss after DP is logged
    assert "train_loss" in metrics, "Should include train_loss (before DP)"
    assert "train_loss_after_dp" in metrics, "Should include train_loss_after_dp"
    assert isinstance(metrics["train_loss_after_dp"], float), "Loss should be a float"
    
    # Loss after DP might be higher due to noise
    print("  ✓ Local training loss after DP application logged")
    print(f"    - Train loss (before DP): {metrics['train_loss']:.4f}")
    print(f"    - Train loss (after DP): {metrics['train_loss_after_dp']:.4f}")
    print(f"    - Loss increase: {metrics['train_loss_after_dp'] - metrics['train_loss']:.4f}")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 80)
    print("DIFFERENTIAL PRIVACY TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        test_dp_config_creation,
        test_gradient_clipping,
        test_noise_addition,
        test_dp_to_updates,
        test_disabled_dp,
        test_flower_client_with_dp,
        test_flower_client_without_dp,
        test_factory_function_with_dp,
        test_privacy_budget_logging,
        test_loss_after_dp,
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
