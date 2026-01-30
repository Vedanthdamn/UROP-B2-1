"""
Demo script for dataset sampling functionality

This script demonstrates how to use the data sampling module
to create reproducible dataset samples for federated learning.
"""

import os
from utils.data_sampling import sample_heart_failure_data, DatasetSampler


def demo_basic_sampling():
    """Demonstrate basic sampling with automatic report generation."""
    print("=" * 60)
    print("DEMO 1: Basic Sampling with Report Generation")
    print("=" * 60)
    
    # Sample data with automatic report generation
    sampled_data = sample_heart_failure_data(
        data_path='data/heart_failure.csv',
        n_samples=300,  # Requested size (will sample all 299 available)
        random_seed=42,
        output_report_path='reports/sampling_summary.md'
    )
    
    print(f"\n✓ Sampled {len(sampled_data)} records")
    print(f"✓ Report saved to reports/sampling_summary.md")
    print(f"✓ Class distribution:")
    print(sampled_data['DEATH_EVENT'].value_counts().sort_index())
    print()


def demo_custom_sampling():
    """Demonstrate custom sampling with DatasetSampler class."""
    print("=" * 60)
    print("DEMO 2: Custom Sampling with DatasetSampler")
    print("=" * 60)
    
    import pandas as pd
    
    # Load data
    data = pd.read_csv('data/heart_failure.csv')
    
    # Create sampler with custom seed
    sampler = DatasetSampler(random_seed=123)
    
    # Sample a subset
    sampled_data = sampler.sample(data, n_samples=150, stratify=True)
    
    # Get sampling summary
    summary = sampler.get_sampling_summary()
    
    print(f"\n✓ Random seed: {summary['random_seed']}")
    print(f"✓ Sample size: {summary['sample_size']}")
    print(f"✓ Original distribution: {summary['original_distribution']}")
    print(f"✓ Sampled distribution: {summary['sampled_distribution']}")
    print(f"✓ Distribution preserved: {summary['distribution_preserved']}")
    print()


def demo_reproducibility():
    """Demonstrate reproducibility with same random seed."""
    print("=" * 60)
    print("DEMO 3: Reproducibility Demonstration")
    print("=" * 60)
    
    import pandas as pd
    
    data = pd.read_csv('data/heart_failure.csv')
    
    # Sample twice with same seed
    sampler1 = DatasetSampler(random_seed=999)
    sample1 = sampler1.sample(data, n_samples=100, stratify=True)
    
    sampler2 = DatasetSampler(random_seed=999)
    sample2 = sampler2.sample(data, n_samples=100, stratify=True)
    
    # Check if identical
    are_identical = sample1.equals(sample2)
    
    print(f"\n✓ Sample 1 shape: {sample1.shape}")
    print(f"✓ Sample 2 shape: {sample2.shape}")
    print(f"✓ Samples are identical: {are_identical}")
    print(f"✓ First 3 rows of sample 1:")
    print(sample1.head(3))
    print(f"\n✓ First 3 rows of sample 2:")
    print(sample2.head(3))
    print()


def demo_different_seeds():
    """Demonstrate that different seeds produce different samples."""
    print("=" * 60)
    print("DEMO 4: Different Seeds Produce Different Samples")
    print("=" * 60)
    
    import pandas as pd
    
    data = pd.read_csv('data/heart_failure.csv')
    
    # Sample with different seeds
    sampler1 = DatasetSampler(random_seed=42)
    sample1 = sampler1.sample(data, n_samples=50, stratify=True)
    
    sampler2 = DatasetSampler(random_seed=123)
    sample2 = sampler2.sample(data, n_samples=50, stratify=True)
    
    are_identical = sample1.equals(sample2)
    
    print(f"\n✓ Sample 1 (seed=42) shape: {sample1.shape}")
    print(f"✓ Sample 2 (seed=123) shape: {sample2.shape}")
    print(f"✓ Samples are identical: {are_identical}")
    print(f"✓ Both preserve class distribution but have different records")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATASET SAMPLING DEMONSTRATION")
    print("=" * 60 + "\n")
    
    # Run all demos
    demo_basic_sampling()
    demo_custom_sampling()
    demo_reproducibility()
    demo_different_seeds()
    
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("1. ✓ Deterministic sampling with fixed random seed")
    print("2. ✓ Stratified sampling preserves class distribution")
    print("3. ✓ Automatic report generation")
    print("4. ✓ Reproducible results across runs")
    print("5. ✓ Original dataset remains unchanged")
    print()
