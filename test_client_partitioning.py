"""
Test suite for client partitioning module

This script validates the ClientDataPartitioner class to ensure:
1. Exactly 5 clients are created
2. Non-IID characteristics (unequal sizes, different class distributions)
3. No overlap between client datasets
4. All samples are used (no data loss)
5. Deterministic and reproducible partitioning
"""

import numpy as np
import pandas as pd
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.client_partitioning import ClientDataPartitioner, partition_for_federated_clients


def test_partition_creates_five_clients():
    """Test that exactly 5 clients are created."""
    print("=" * 60)
    print("TEST 1: Five Clients Created")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create partitioner and partition
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=42)
    client_datasets = partitioner.partition(df)
    
    # Verify 5 clients
    assert len(client_datasets) == 5, f"Expected 5 clients, got {len(client_datasets)}"
    
    print(f"✓ Created {len(client_datasets)} clients")
    print(f"✓ Five clients test: PASSED\n")
    return True


def test_no_sample_overlap():
    """Test that there is no overlap between client datasets."""
    print("=" * 60)
    print("TEST 2: No Sample Overlap")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create partitioner and partition
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=42)
    client_datasets = partitioner.partition(df)
    
    # Check for overlap by examining actual data
    seen_samples = set()
    for i, client_data in enumerate(client_datasets):
        for _, row in client_data.iterrows():
            sample_id = tuple(row.values)
            assert sample_id not in seen_samples, f"Duplicate sample found in client {i}"
            seen_samples.add(sample_id)
    
    # Also test the built-in method
    assert partitioner.verify_no_overlap(), "verify_no_overlap() returned False"
    
    print(f"✓ Total unique samples: {len(seen_samples)}")
    print(f"✓ Expected samples: {len(df)}")
    print(f"✓ No overlap test: PASSED\n")
    return True


def test_all_samples_used():
    """Test that all samples from the original dataset are used."""
    print("=" * 60)
    print("TEST 3: All Samples Used")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    original_size = len(df)
    
    # Create partitioner and partition
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=42)
    client_datasets = partitioner.partition(df)
    
    # Count total samples across all clients
    total_samples = sum(len(client_data) for client_data in client_datasets)
    
    assert total_samples == original_size, \
        f"Expected {original_size} samples, got {total_samples}"
    
    print(f"✓ Original dataset: {original_size} samples")
    print(f"✓ Partitioned total: {total_samples} samples")
    print(f"✓ All samples used test: PASSED\n")
    return True


def test_non_iid_unequal_sizes():
    """Test that client datasets have unequal sizes (non-IID)."""
    print("=" * 60)
    print("TEST 4: Non-IID Unequal Sizes")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create partitioner and partition
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=42)
    client_datasets = partitioner.partition(df)
    
    # Get sizes
    sizes = [len(client_data) for client_data in client_datasets]
    
    # Check that sizes are not all equal
    assert len(set(sizes)) > 1, "All client datasets have the same size (IID)"
    
    # Check that there's reasonable variation
    std_dev = np.std(sizes)
    assert std_dev > 0, "No variation in client sizes"
    
    print(f"✓ Client sizes: {sizes}")
    print(f"✓ Min size: {min(sizes)}")
    print(f"✓ Max size: {max(sizes)}")
    print(f"✓ Standard deviation: {std_dev:.2f}")
    print(f"✓ Unequal sizes test: PASSED\n")
    return True


def test_non_iid_different_class_distributions():
    """Test that client datasets have different class distributions (non-IID)."""
    print("=" * 60)
    print("TEST 5: Non-IID Different Class Distributions")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create partitioner and partition
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=42)
    client_datasets = partitioner.partition(df)
    
    # Get class 0 proportions for each client
    class_0_proportions = []
    for i, client_data in enumerate(client_datasets):
        class_dist = client_data['DEATH_EVENT'].value_counts().to_dict()
        total = len(client_data)
        class_0_prop = class_dist.get(0, 0) / total * 100
        class_0_proportions.append(class_0_prop)
        print(f"  Client {i}: Class 0 = {class_0_prop:.1f}%")
    
    # Check that proportions are not all equal
    assert len(set(class_0_proportions)) > 1, \
        "All clients have the same class distribution (IID)"
    
    # Check that there's significant variation (std dev > 5%)
    std_dev = np.std(class_0_proportions)
    assert std_dev > 5.0, \
        f"Insufficient variation in class distributions (std dev = {std_dev:.2f}%)"
    
    print(f"\n✓ Class 0 proportion range: {min(class_0_proportions):.1f}% - "
          f"{max(class_0_proportions):.1f}%")
    print(f"✓ Standard deviation: {std_dev:.2f}%")
    print(f"✓ Different class distributions test: PASSED\n")
    return True


def test_deterministic_partitioning():
    """Test that partitioning is deterministic with fixed random seed."""
    print("=" * 60)
    print("TEST 6: Deterministic Partitioning")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Partition twice with same seed
    partitioner1 = ClientDataPartitioner(n_clients=5, random_seed=999)
    client_datasets1 = partitioner1.partition(df.copy())
    
    partitioner2 = ClientDataPartitioner(n_clients=5, random_seed=999)
    client_datasets2 = partitioner2.partition(df.copy())
    
    # Check that partitions are identical
    for i in range(5):
        assert client_datasets1[i].equals(client_datasets2[i]), \
            f"Client {i} datasets differ with same seed"
    
    print(f"✓ Partition 1 and Partition 2 are identical")
    print(f"✓ Deterministic partitioning test: PASSED\n")
    return True


def test_convenience_function():
    """Test the convenience function partition_for_federated_clients."""
    print("=" * 60)
    print("TEST 7: Convenience Function")
    print("=" * 60)
    
    # Use convenience function
    client_datasets = partition_for_federated_clients(
        'data/heart_failure.csv',
        n_clients=5,
        random_seed=42
    )
    
    assert len(client_datasets) == 5, "Convenience function did not create 5 clients"
    
    total_samples = sum(len(cd) for cd in client_datasets)
    original_size = len(pd.read_csv('data/heart_failure.csv'))
    
    assert total_samples == original_size, \
        "Convenience function did not partition all samples"
    
    print(f"✓ Created {len(client_datasets)} clients")
    print(f"✓ Total samples: {total_samples}")
    print(f"✓ Convenience function test: PASSED\n")
    return True


def test_partition_info():
    """Test that partition info is correctly computed."""
    print("=" * 60)
    print("TEST 8: Partition Info")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create partitioner and partition
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=42)
    client_datasets = partitioner.partition(df)
    
    # Get partition info
    info = partitioner.get_partition_info()
    
    assert 'n_clients' in info, "Missing n_clients in partition info"
    assert 'random_seed' in info, "Missing random_seed in partition info"
    assert 'clients' in info, "Missing clients in partition info"
    assert len(info['clients']) == 5, "Expected 5 client info entries"
    
    # Check each client info
    for i, client_info in enumerate(info['clients']):
        assert 'client_id' in client_info, f"Missing client_id in client {i}"
        assert 'n_samples' in client_info, f"Missing n_samples in client {i}"
        assert 'class_distribution' in client_info, f"Missing class_distribution in client {i}"
        assert 'class_proportions' in client_info, f"Missing class_proportions in client {i}"
    
    print(f"✓ Partition info contains all required fields")
    print(f"✓ Partition info test: PASSED\n")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("CLIENT PARTITIONING TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_partition_creates_five_clients,
        test_no_sample_overlap,
        test_all_samples_used,
        test_non_iid_unequal_sizes,
        test_non_iid_different_class_distributions,
        test_deterministic_partitioning,
        test_convenience_function,
        test_partition_info,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {str(e)}\n")
        except Exception as e:
            failed += 1
            print(f"✗ TEST ERROR: {test.__name__}")
            print(f"  Error: {str(e)}\n")
    
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
