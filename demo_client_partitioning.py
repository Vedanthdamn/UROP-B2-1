"""
Demo script for client partitioning functionality

This script demonstrates how to use the client partitioning module
to create non-IID partitions for federated learning.
"""

import os
from utils.client_partitioning import partition_for_federated_clients, ClientDataPartitioner
import pandas as pd


def demo_basic_partitioning():
    """Demonstrate basic partitioning with automatic report generation."""
    print("=" * 60)
    print("DEMO 1: Basic Partitioning with Report Generation")
    print("=" * 60)
    
    # Partition data with automatic report generation
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=5,
        random_seed=42,
        output_report_path='reports/client_partition_summary.md'
    )
    
    print(f"\n✓ Partitioned into {len(client_datasets)} clients")
    print(f"✓ Report saved to reports/client_partition_summary.md")
    print(f"\n✓ Client sizes:")
    for i, client_data in enumerate(client_datasets):
        class_dist = client_data['DEATH_EVENT'].value_counts().to_dict()
        print(f"  Client {i}: {len(client_data)} samples "
              f"(Class 0: {class_dist.get(0, 0)}, Class 1: {class_dist.get(1, 0)})")
    print()


def demo_advanced_partitioning():
    """Demonstrate advanced partitioning with ClientDataPartitioner class."""
    print("=" * 60)
    print("DEMO 2: Advanced Partitioning with ClientDataPartitioner")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/heart_failure.csv')
    
    # Create partitioner with custom parameters
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=123)
    
    # Perform partitioning
    client_datasets = partitioner.partition(data)
    
    # Get partition info
    info = partitioner.get_partition_info()
    
    print(f"\n✓ Number of clients: {info['n_clients']}")
    print(f"✓ Random seed: {info['random_seed']}")
    print(f"\n✓ Client Statistics:")
    for client in info['clients']:
        print(f"  Client {client['client_id']}: {client['n_samples']} samples")
        for cls, prop in sorted(client['class_proportions'].items()):
            print(f"    Class {cls}: {prop:.1f}%")
    
    # Verify no overlap
    no_overlap = partitioner.verify_no_overlap()
    print(f"\n✓ No sample overlap: {no_overlap}")
    print()


def demo_non_iid_characteristics():
    """Demonstrate non-IID characteristics of the partition."""
    print("=" * 60)
    print("DEMO 3: Non-IID Characteristics")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/heart_failure.csv')
    
    # Create partitioner
    partitioner = ClientDataPartitioner(n_clients=5, random_seed=42)
    client_datasets = partitioner.partition(data)
    
    # Show unequal sizes
    sizes = [len(cd) for cd in client_datasets]
    print(f"\n✓ Unequal Sample Sizes (Non-IID):")
    print(f"  Sizes: {sizes}")
    print(f"  Min: {min(sizes)}, Max: {max(sizes)}")
    print(f"  Range: {max(sizes) - min(sizes)} samples")
    
    # Show different class distributions
    print(f"\n✓ Different Class Distributions (Non-IID):")
    for i, client_data in enumerate(client_datasets):
        class_dist = client_data['DEATH_EVENT'].value_counts().to_dict()
        total = len(client_data)
        class_0_pct = class_dist.get(0, 0) / total * 100
        class_1_pct = class_dist.get(1, 0) / total * 100
        print(f"  Client {i}: Class 0={class_0_pct:.1f}%, Class 1={class_1_pct:.1f}%")
    print()


def demo_reproducibility():
    """Demonstrate reproducibility with same random seed."""
    print("=" * 60)
    print("DEMO 4: Reproducibility")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/heart_failure.csv')
    
    # Partition twice with same seed
    partitioner1 = ClientDataPartitioner(n_clients=5, random_seed=999)
    clients1 = partitioner1.partition(data.copy())
    
    partitioner2 = ClientDataPartitioner(n_clients=5, random_seed=999)
    clients2 = partitioner2.partition(data.copy())
    
    # Check if partitions are identical
    identical = all(
        clients1[i].equals(clients2[i]) 
        for i in range(5)
    )
    
    print(f"\n✓ Partition 1 clients: {len(clients1)}")
    print(f"✓ Partition 2 clients: {len(clients2)}")
    print(f"✓ Partitions are identical: {identical}")
    print(f"\n✓ First 3 rows of Client 0 from Partition 1:")
    print(clients1[0].head(3))
    print(f"\n✓ First 3 rows of Client 0 from Partition 2:")
    print(clients2[0].head(3))
    print()


def demo_different_seeds():
    """Demonstrate that different seeds produce different partitions."""
    print("=" * 60)
    print("DEMO 5: Different Seeds Produce Different Partitions")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/heart_failure.csv')
    
    # Partition with different seeds
    partitioner1 = ClientDataPartitioner(n_clients=5, random_seed=42)
    clients1 = partitioner1.partition(data.copy())
    
    partitioner2 = ClientDataPartitioner(n_clients=5, random_seed=123)
    clients2 = partitioner2.partition(data.copy())
    
    # Compare sizes and distributions
    print(f"\n✓ Seed 42 client sizes: {[len(c) for c in clients1]}")
    print(f"✓ Seed 123 client sizes: {[len(c) for c in clients2]}")
    
    identical = all(
        clients1[i].equals(clients2[i]) 
        for i in range(5)
    )
    print(f"\n✓ Partitions are identical: {identical}")
    print(f"✓ Both maintain non-IID characteristics but with different samples")
    print()


def demo_federated_learning_simulation():
    """Demonstrate how to use partitions in a federated learning simulation."""
    print("=" * 60)
    print("DEMO 6: Federated Learning Simulation Setup")
    print("=" * 60)
    
    # Partition data for federated learning
    client_datasets = partition_for_federated_clients(
        'data/heart_failure.csv',
        n_clients=5,
        random_seed=42
    )
    
    print(f"\n✓ Simulating federated learning with {len(client_datasets)} hospitals")
    print(f"\n✓ Hospital Setup:")
    
    for i, client_data in enumerate(client_datasets):
        print(f"\n  Hospital {i}:")
        print(f"    - Patient records: {len(client_data)}")
        print(f"    - Features: {client_data.shape[1] - 1}")  # Exclude target
        print(f"    - Target column: DEATH_EVENT")
        
        # Show sample data characteristics
        class_dist = client_data['DEATH_EVENT'].value_counts().to_dict()
        print(f"    - Class distribution:")
        for cls, count in sorted(class_dist.items()):
            print(f"      * Class {cls}: {count} samples ({count/len(client_data)*100:.1f}%)")
    
    print(f"\n✓ Federated Learning Ready:")
    print(f"  - Each hospital has its own local dataset")
    print(f"  - Non-IID characteristics simulate real-world scenarios")
    print(f"  - No data overlap between hospitals")
    print(f"  - Ready for federated training experiments")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CLIENT PARTITIONING DEMONSTRATION")
    print("=" * 60 + "\n")
    
    # Run all demos
    demo_basic_partitioning()
    demo_advanced_partitioning()
    demo_non_iid_characteristics()
    demo_reproducibility()
    demo_different_seeds()
    demo_federated_learning_simulation()
    
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("1. ✓ Non-IID partitioning with unequal sample sizes")
    print("2. ✓ Different class distributions per client")
    print("3. ✓ No sample overlap between clients")
    print("4. ✓ Deterministic and reproducible partitioning")
    print("5. ✓ Automatic report generation")
    print("6. ✓ Ready for federated learning simulation")
    print()
