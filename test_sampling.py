"""
Test suite for data sampling module

This script validates the DatasetSampler class to ensure:
1. Deterministic sampling with fixed random seed
2. Stratified sampling preserves class distribution
3. Proper handling of edge cases (sample size > dataset size)
4. Report generation works correctly
5. Reproducibility across multiple runs
"""

import numpy as np
import pandas as pd
import os
import tempfile
import sys

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_sampling import DatasetSampler, sample_heart_failure_data


def test_deterministic_sampling():
    """Test that sampling is deterministic with fixed random seed."""
    print("=" * 60)
    print("TEST 1: Deterministic Sampling")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Sample twice with same seed
    sampler1 = DatasetSampler(random_seed=42)
    sampled1 = sampler1.sample(df, n_samples=100, stratify=True)
    
    sampler2 = DatasetSampler(random_seed=42)
    sampled2 = sampler2.sample(df, n_samples=100, stratify=True)
    
    # Check that samples are identical
    assert sampled1.equals(sampled2), "Samples with same seed should be identical"
    
    print(f"✓ Sample 1 shape: {sampled1.shape}")
    print(f"✓ Sample 2 shape: {sampled2.shape}")
    print(f"✓ Samples are identical")
    print(f"✓ Deterministic sampling: PASSED\n")
    return True


def test_stratified_sampling():
    """Test that stratified sampling preserves class distribution."""
    print("=" * 60)
    print("TEST 2: Stratified Sampling")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Calculate original proportions
    original_counts = df['DEATH_EVENT'].value_counts()
    original_prop_0 = original_counts[0] / len(df)
    original_prop_1 = original_counts[1] / len(df)
    
    print(f"Original proportions - Class 0: {original_prop_0:.4f}, Class 1: {original_prop_1:.4f}")
    
    # Perform stratified sampling
    sampler = DatasetSampler(random_seed=42)
    sampled = sampler.sample(df, n_samples=200, stratify=True)
    
    # Calculate sampled proportions
    sampled_counts = sampled['DEATH_EVENT'].value_counts()
    sampled_prop_0 = sampled_counts[0] / len(sampled)
    sampled_prop_1 = sampled_counts[1] / len(sampled)
    
    print(f"Sampled proportions - Class 0: {sampled_prop_0:.4f}, Class 1: {sampled_prop_1:.4f}")
    
    # Check that proportions are similar (within 5% tolerance)
    assert abs(original_prop_0 - sampled_prop_0) < 0.05, "Class 0 proportion not preserved"
    assert abs(original_prop_1 - sampled_prop_1) < 0.05, "Class 1 proportion not preserved"
    
    print(f"✓ Class distribution preserved (within 5% tolerance)")
    print(f"✓ Stratified sampling: PASSED\n")
    return True


def test_sample_size_exceeds_dataset():
    """Test handling when requested sample size exceeds dataset size."""
    print("=" * 60)
    print("TEST 3: Sample Size Exceeds Dataset")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    original_size = len(df)
    
    # Request more samples than available
    sampler = DatasetSampler(random_seed=42)
    sampled = sampler.sample(df, n_samples=500, stratify=True)
    
    # Should return all available records
    assert len(sampled) == original_size, f"Expected {original_size} records, got {len(sampled)}"
    
    # Check that all classes are present with correct counts
    original_counts = df['DEATH_EVENT'].value_counts().to_dict()
    sampled_counts = sampled['DEATH_EVENT'].value_counts().to_dict()
    for class_label in original_counts:
        assert class_label in sampled_counts, f"Class {class_label} missing from sample"
        assert sampled_counts[class_label] == original_counts[class_label], \
            f"Class {class_label} count mismatch: {sampled_counts[class_label]} != {original_counts[class_label]}"
    
    print(f"✓ Requested: 500 records")
    print(f"✓ Available: {original_size} records")
    print(f"✓ Returned: {len(sampled)} records")
    print(f"✓ All records returned when sample size exceeds dataset")
    print(f"✓ Sample size handling: PASSED\n")
    return True


def test_sampling_summary():
    """Test that sampling summary is generated correctly."""
    print("=" * 60)
    print("TEST 4: Sampling Summary")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Perform sampling
    sampler = DatasetSampler(random_seed=123)
    sampled = sampler.sample(df, n_samples=150, stratify=True)
    
    # Get summary
    summary = sampler.get_sampling_summary()
    
    # Validate summary structure
    assert 'random_seed' in summary, "Summary missing random_seed"
    assert 'sample_size' in summary, "Summary missing sample_size"
    assert 'original_distribution' in summary, "Summary missing original_distribution"
    assert 'sampled_distribution' in summary, "Summary missing sampled_distribution"
    assert 'distribution_preserved' in summary, "Summary missing distribution_preserved"
    
    # Validate values
    assert summary['random_seed'] == 123, f"Expected seed 123, got {summary['random_seed']}"
    assert summary['sample_size'] == 150, f"Expected size 150, got {summary['sample_size']}"
    assert isinstance(summary['distribution_preserved'], bool), "distribution_preserved should be bool"
    
    print(f"✓ Summary contains all required fields")
    print(f"✓ Random seed: {summary['random_seed']}")
    print(f"✓ Sample size: {summary['sample_size']}")
    print(f"✓ Distribution preserved: {summary['distribution_preserved']}")
    print(f"✓ Sampling summary: PASSED\n")
    return True


def test_report_generation():
    """Test that markdown report is generated correctly."""
    print("=" * 60)
    print("TEST 5: Report Generation")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Create temporary file for report
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Perform sampling and generate report
        sampler = DatasetSampler(random_seed=42)
        sampled = sampler.sample(df, n_samples=100, stratify=True)
        sampler.generate_report(tmp_path)
        
        # Check that report file was created
        assert os.path.exists(tmp_path), "Report file not created"
        
        # Read and validate report content
        with open(tmp_path, 'r') as f:
            report_content = f.read()
        
        # Check for key sections
        assert '# Dataset Sampling Summary' in report_content, "Report missing title"
        assert '## Overview' in report_content, "Report missing Overview section"
        assert '## Class Distribution' in report_content, "Report missing Class Distribution section"
        assert '## Reproducibility' in report_content, "Report missing Reproducibility section"
        assert 'Random Seed' in report_content, "Report missing random seed"
        
        print(f"✓ Report file created at {tmp_path}")
        print(f"✓ Report contains all required sections")
        print(f"✓ Report length: {len(report_content)} characters")
        print(f"✓ Report generation: PASSED\n")
        return True
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_different_random_seeds():
    """Test that different random seeds produce different samples."""
    print("=" * 60)
    print("TEST 6: Different Random Seeds")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Sample with different seeds
    sampler1 = DatasetSampler(random_seed=42)
    sampled1 = sampler1.sample(df, n_samples=100, stratify=True)
    
    sampler2 = DatasetSampler(random_seed=123)
    sampled2 = sampler2.sample(df, n_samples=100, stratify=True)
    
    # Samples should be different
    assert not sampled1.equals(sampled2), "Samples with different seeds should be different"
    
    # But same size and column structure
    assert sampled1.shape == sampled2.shape, "Samples should have same shape"
    assert list(sampled1.columns) == list(sampled2.columns), "Samples should have same columns"
    
    print(f"✓ Seed 42 sample shape: {sampled1.shape}")
    print(f"✓ Seed 123 sample shape: {sampled2.shape}")
    print(f"✓ Samples are different")
    print(f"✓ Different random seeds: PASSED\n")
    return True


def test_convenience_function():
    """Test the convenience function sample_heart_failure_data."""
    print("=" * 60)
    print("TEST 7: Convenience Function")
    print("=" * 60)
    
    # Create temporary report path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Use convenience function
        sampled = sample_heart_failure_data(
            'data/heart_failure.csv',
            n_samples=150,
            random_seed=42,
            output_report_path=tmp_path
        )
        
        # Validate results
        assert isinstance(sampled, pd.DataFrame), "Should return DataFrame"
        assert len(sampled) == 150, f"Expected 150 records, got {len(sampled)}"
        assert 'DEATH_EVENT' in sampled.columns, "Missing target column"
        assert os.path.exists(tmp_path), "Report not generated"
        
        print(f"✓ Function returned DataFrame with {len(sampled)} records")
        print(f"✓ Report generated at {tmp_path}")
        print(f"✓ Convenience function: PASSED\n")
        return True
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_target_column_validation():
    """Test that target column validation works correctly."""
    print("=" * 60)
    print("TEST 8: Target Column Validation")
    print("=" * 60)
    
    # Create test data with different target column
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Try to sample with wrong target column name
    sampler = DatasetSampler(random_seed=42, target_column='DEATH_EVENT')
    
    try:
        sampled = sampler.sample(df, n_samples=3)
        assert False, "Should raise ValueError for missing target column"
    except ValueError as e:
        assert 'not found in data' in str(e), "Error message should mention missing column"
        print(f"✓ Correctly raises ValueError for missing target column")
    
    # Now with correct target column
    sampler = DatasetSampler(random_seed=42, target_column='target')
    sampled = sampler.sample(df, n_samples=3)
    assert len(sampled) == 3, "Should successfully sample with correct target column"
    print(f"✓ Successfully samples with correct target column")
    print(f"✓ Target column validation: PASSED\n")
    return True


def test_reproducibility_across_runs():
    """Test reproducibility by running sampling multiple times."""
    print("=" * 60)
    print("TEST 9: Reproducibility Across Multiple Runs")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Perform sampling 3 times with same seed
    results = []
    for i in range(3):
        sampler = DatasetSampler(random_seed=999)
        sampled = sampler.sample(df, n_samples=50, stratify=True)
        results.append(sampled)
    
    # All results should be identical
    for i in range(1, len(results)):
        assert results[0].equals(results[i]), f"Run {i+1} produced different results"
    
    print(f"✓ Performed 3 independent sampling runs")
    print(f"✓ All runs produced identical results")
    print(f"✓ Reproducibility: PASSED\n")
    return True


def test_small_sample_size():
    """Test sampling with very small sample size."""
    print("=" * 60)
    print("TEST 10: Small Sample Size")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('data/heart_failure.csv')
    
    # Sample very small subset
    sampler = DatasetSampler(random_seed=42)
    sampled = sampler.sample(df, n_samples=10, stratify=True)
    
    # Validate
    assert len(sampled) == 10, f"Expected 10 records, got {len(sampled)}"
    assert 'DEATH_EVENT' in sampled.columns, "Target column missing"
    
    # Check that both classes might be present (or at least one)
    class_counts = sampled['DEATH_EVENT'].value_counts()
    assert len(class_counts) > 0, "No classes in sample"
    
    print(f"✓ Successfully sampled {len(sampled)} records")
    print(f"✓ Class distribution: {class_counts.to_dict()}")
    print(f"✓ Small sample size: PASSED\n")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("DATA SAMPLING TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_deterministic_sampling,
        test_stratified_sampling,
        test_sample_size_exceeds_dataset,
        test_sampling_summary,
        test_report_generation,
        test_different_random_seeds,
        test_convenience_function,
        test_target_column_validation,
        test_reproducibility_across_runs,
        test_small_sample_size,
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
