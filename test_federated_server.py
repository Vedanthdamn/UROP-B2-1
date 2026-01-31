"""
Test suite for Flower federated server implementation.

This test suite validates the Flower server for federated learning:
1. Server initialization
2. FedAvg strategy creation
3. FedProx strategy creation
4. Metrics aggregation
5. Simulation mode training

Usage:
    python test_federated_server.py
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from federated import (
    FederatedServer,
    create_federated_server,
    start_server_simulation,
    create_flower_client,
    create_dp_config
)
from models import get_primary_model
from utils.preprocessing import create_preprocessing_pipeline
from utils.client_partitioning import partition_for_federated_clients


def test_server_initialization():
    """Test federated server initialization."""
    print("Test 1: Server Initialization")
    
    # Create server with FedAvg
    server = FederatedServer(
        strategy_name="fedavg",
        num_rounds=5,
        min_available_clients=2,
        input_shape=(1, 12)
    )
    
    assert server is not None, "Server should not be None"
    assert server.strategy_name == "fedavg", "Strategy name mismatch"
    assert server.num_rounds == 5, "Number of rounds mismatch"
    assert server.global_model is not None, "Global model should not be None"
    assert server.strategy is not None, "Strategy should not be None"
    
    print("  ✓ Server initialization successful (FedAvg)")


def test_fedprox_initialization():
    """Test FedProx strategy initialization."""
    print("Test 2: FedProx Strategy Initialization")
    
    # Create server with FedProx
    server = FederatedServer(
        strategy_name="fedprox",
        num_rounds=5,
        min_available_clients=2,
        input_shape=(1, 12),
        proximal_mu=0.1
    )
    
    assert server is not None, "Server should not be None"
    assert server.strategy_name == "fedprox", "Strategy name mismatch"
    assert server.proximal_mu == 0.1, "Proximal mu mismatch"
    
    print("  ✓ FedProx strategy initialization successful")


def test_create_federated_server():
    """Test factory function for server creation."""
    print("Test 3: Factory Function (create_federated_server)")
    
    # Create server using factory function
    server = create_federated_server(
        strategy="fedavg",
        num_rounds=10,
        min_clients=3,
        input_shape=(1, 12)
    )
    
    assert server is not None, "Server should not be None"
    assert server.num_rounds == 10, "Number of rounds mismatch"
    assert server.min_available_clients == 3, "Min clients mismatch"
    
    print("  ✓ Factory function successful")


def test_metrics_aggregation():
    """Test metrics aggregation functions."""
    print("Test 4: Metrics Aggregation")
    
    # Create server
    server = FederatedServer(
        strategy_name="fedavg",
        num_rounds=5,
        min_available_clients=2
    )
    
    # Simulate client metrics
    client_metrics = [
        (50, {"train_loss": 0.7, "train_accuracy": 0.6}),
        (30, {"train_loss": 0.8, "train_accuracy": 0.5}),
        (20, {"train_loss": 0.6, "train_accuracy": 0.7})
    ]
    
    # Aggregate metrics
    aggregated = server._aggregate_fit_metrics(client_metrics)
    
    assert "train_loss" in aggregated, "train_loss missing from aggregated metrics"
    assert "train_accuracy" in aggregated, "train_accuracy missing from aggregated metrics"
    assert "num_clients" in aggregated, "num_clients missing from aggregated metrics"
    assert aggregated["num_clients"] == 3, "Number of clients mismatch"
    assert aggregated["total_samples"] == 100, "Total samples mismatch"
    
    # Check weighted average
    expected_loss = (0.7 * 50 + 0.8 * 30 + 0.6 * 20) / 100
    assert abs(aggregated["train_loss"] - expected_loss) < 0.001, "Loss aggregation incorrect"
    
    print("  ✓ Metrics aggregation successful")


def test_simulation_mode():
    """Test federated learning in simulation mode."""
    print("Test 5: Simulation Mode Training")
    
    # Prepare data
    print("  - Loading and partitioning data...")
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=3,
        random_seed=42
    )
    
    # Create preprocessor
    print("  - Creating preprocessing pipeline...")
    data = pd.read_csv('data/heart_failure.csv')
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    # Define client factory
    def client_fn(cid: str):
        client_id = int(cid)
        return create_flower_client(
            client_data=client_datasets[client_id],
            preprocessor=preprocessor,
            val_split=0.2,
            epochs_per_round=2,  # Small for testing
            batch_size=32,
            client_id=f"hospital_{client_id}",
            random_seed=42
        )
    
    # Run simulation
    print("  - Running federated training (2 rounds)...")
    try:
        history = start_server_simulation(
            client_fn=client_fn,
            num_clients=3,
            strategy="fedavg",
            num_rounds=2,
            input_shape=(1, 12)
        )
        
        assert history is not None, "History should not be None"
        print("  ✓ Simulation mode training successful")
        return True
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        return False


def test_simulation_with_dp():
    """Test federated learning with differential privacy."""
    print("Test 6: Simulation with Differential Privacy")
    
    # Prepare data
    print("  - Loading and partitioning data...")
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=3,
        random_seed=42
    )
    
    # Create preprocessor
    print("  - Creating preprocessing pipeline...")
    data = pd.read_csv('data/heart_failure.csv')
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    # Create DP config
    dp_config = create_dp_config(
        epsilon=1.0,
        delta=1e-5,
        l2_norm_clip=1.0,
        enabled=True
    )
    
    # Define client factory with DP
    def client_fn(cid: str):
        client_id = int(cid)
        return create_flower_client(
            client_data=client_datasets[client_id],
            preprocessor=preprocessor,
            val_split=0.2,
            epochs_per_round=2,  # Small for testing
            batch_size=32,
            client_id=f"hospital_{client_id}",
            random_seed=42,
            dp_config=dp_config
        )
    
    # Run simulation
    print("  - Running federated training with DP (2 rounds)...")
    try:
        history = start_server_simulation(
            client_fn=client_fn,
            num_clients=3,
            strategy="fedavg",
            num_rounds=2,
            input_shape=(1, 12)
        )
        
        assert history is not None, "History should not be None"
        print("  ✓ Simulation with DP successful")
        return True
    except Exception as e:
        print(f"  ✗ Simulation with DP failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("FEDERATED SERVER TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        ("Server Initialization", test_server_initialization),
        ("FedProx Strategy", test_fedprox_initialization),
        ("Factory Function", test_create_federated_server),
        ("Metrics Aggregation", test_metrics_aggregation),
        ("Simulation Mode", test_simulation_mode),
        ("Simulation with DP", test_simulation_with_dp),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            failed += 1
        print()
    
    # Print summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 80 + "\n")
    
    if failed > 0:
        print("❌ Some tests failed")
        sys.exit(1)
    else:
        print("✅ All tests passed")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
