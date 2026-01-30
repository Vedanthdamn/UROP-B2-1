"""
Deterministic Dataset Sampling for Federated Learning

This module provides deterministic sampling functionality to create reproducible
training subsets for federated learning experiments. It ensures:
- Fixed random seed for reproducibility
- Stratified sampling to preserve class distribution
- Detailed logging of sampling parameters and results

Author: Federated Learning Medical AI Project
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetSampler:
    """
    Deterministic dataset sampler with stratified sampling support.
    
    This class provides reproducible dataset sampling for federated learning
    experiments. It uses a fixed random seed and stratified sampling to ensure
    the sampled dataset maintains the original class distribution.
    
    Attributes:
        random_seed (int): Fixed random seed for reproducibility
        sample_size (int): Number of records to sample
        target_column (str): Name of the target column for stratification
        original_distribution (dict): Class distribution in original dataset
        sampled_distribution (dict): Class distribution in sampled dataset
    """
    
    def __init__(self, random_seed: int = 42, target_column: str = 'DEATH_EVENT'):
        """
        Initialize the dataset sampler.
        
        Args:
            random_seed (int): Random seed for reproducibility. Default is 42.
            target_column (str): Name of the target column. Default is 'DEATH_EVENT'.
        """
        self.random_seed = random_seed
        self.target_column = target_column
        self.sample_size = None
        self.original_distribution = None
        self.sampled_distribution = None
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        logger.info(f"DatasetSampler initialized with random_seed={random_seed}")
    
    def sample(self, 
               data: pd.DataFrame, 
               n_samples: int,
               stratify: bool = True) -> pd.DataFrame:
        """
        Sample data with stratification to preserve class distribution.
        
        This method performs deterministic sampling using the configured random seed.
        If stratified sampling is enabled, it attempts to maintain the original
        class distribution in the sampled dataset.
        
        Args:
            data (pd.DataFrame): Original dataset to sample from
            n_samples (int): Number of samples to draw
            stratify (bool): Whether to use stratified sampling. Default is True.
        
        Returns:
            pd.DataFrame: Sampled dataset
        
        Raises:
            ValueError: If target column is not found or sample size is invalid
        """
        self.sample_size = n_samples
        
        # Validate inputs
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data. "
                           f"Available columns: {data.columns.tolist()}")
        
        if n_samples <= 0:
            raise ValueError(f"Sample size must be positive, got {n_samples}")
        
        if n_samples > len(data):
            logger.warning(f"Requested sample size ({n_samples}) exceeds dataset size "
                         f"({len(data)}). Sampling all available records.")
            n_samples = len(data)
        
        # Calculate original class distribution
        self.original_distribution = data[self.target_column].value_counts().to_dict()
        logger.info(f"Original dataset size: {len(data)}")
        logger.info(f"Original class distribution: {self.original_distribution}")
        
        # Perform sampling
        if stratify and n_samples < len(data):
            # Stratified sampling to preserve class distribution
            sampled_data = self._stratified_sample(data, n_samples)
        else:
            # If sampling all records or stratification disabled, just return all data
            # (shuffled for consistency)
            sampled_data = data.sample(n=n_samples, random_state=self.random_seed)
        
        # Calculate sampled class distribution
        self.sampled_distribution = sampled_data[self.target_column].value_counts().to_dict()
        logger.info(f"Sampled dataset size: {len(sampled_data)}")
        logger.info(f"Sampled class distribution: {self.sampled_distribution}")
        
        return sampled_data.reset_index(drop=True)
    
    def _stratified_sample(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        Perform stratified sampling to preserve class distribution.
        
        Args:
            data (pd.DataFrame): Dataset to sample from
            n_samples (int): Number of samples to draw
        
        Returns:
            pd.DataFrame: Stratified sample
        """
        # Calculate sample size for each class proportionally
        class_counts = data[self.target_column].value_counts()
        total_size = len(data)
        
        samples_per_class = {}
        remaining_samples = n_samples
        
        # Allocate samples proportionally to each class
        for class_label in sorted(class_counts.index):
            class_size = class_counts[class_label]
            proportion = class_size / total_size
            
            # Calculate proportional sample size
            if class_label == sorted(class_counts.index)[-1]:
                # Last class gets remaining samples to ensure exact total
                samples_per_class[class_label] = remaining_samples
            else:
                samples_per_class[class_label] = int(np.round(proportion * n_samples))
                remaining_samples -= samples_per_class[class_label]
        
        logger.info(f"Stratified sampling plan: {samples_per_class}")
        
        # Sample from each class
        sampled_dfs = []
        for class_label, n_class_samples in samples_per_class.items():
            class_data = data[data[self.target_column] == class_label]
            
            # Ensure we don't try to sample more than available
            n_class_samples = min(n_class_samples, len(class_data))
            
            sampled_class = class_data.sample(
                n=n_class_samples, 
                random_state=self.random_seed + int(class_label)
            )
            sampled_dfs.append(sampled_class)
        
        # Combine all class samples
        sampled_data = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the combined sample for good measure
        sampled_data = sampled_data.sample(
            frac=1.0, 
            random_state=self.random_seed
        ).reset_index(drop=True)
        
        return sampled_data
    
    def get_sampling_summary(self) -> dict:
        """
        Get a summary of the sampling operation.
        
        Returns:
            dict: Dictionary containing sampling details including:
                - random_seed: The random seed used
                - sample_size: Number of samples drawn
                - original_distribution: Original class distribution
                - sampled_distribution: Sampled class distribution
                - distribution_preserved: Whether distribution was preserved
        
        Raises:
            RuntimeError: If no sampling has been performed yet
        """
        if self.sample_size is None:
            raise RuntimeError("No sampling performed yet. Call sample() first.")
        
        # Calculate distribution preservation metrics
        distribution_preserved = self._check_distribution_preserved()
        
        return {
            'random_seed': self.random_seed,
            'sample_size': self.sample_size,
            'original_distribution': self.original_distribution,
            'sampled_distribution': self.sampled_distribution,
            'distribution_preserved': distribution_preserved
        }
    
    def _check_distribution_preserved(self) -> bool:
        """
        Check if the class distribution was preserved in sampling.
        
        Returns:
            bool: True if distribution is approximately preserved, False otherwise
        """
        if self.original_distribution is None or self.sampled_distribution is None:
            return False
        
        # Calculate proportions
        total_original = sum(self.original_distribution.values())
        total_sampled = sum(self.sampled_distribution.values())
        
        # Check if proportions are similar (within 5% tolerance)
        for class_label in self.original_distribution.keys():
            original_prop = self.original_distribution[class_label] / total_original
            sampled_prop = self.sampled_distribution.get(class_label, 0) / total_sampled
            
            # Allow 5% deviation
            if abs(original_prop - sampled_prop) > 0.05:
                return False
        
        return True
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate a markdown report of the sampling operation.
        
        Args:
            output_path (str): Path where the report should be saved
        
        Raises:
            RuntimeError: If no sampling has been performed yet
        """
        if self.sample_size is None:
            raise RuntimeError("No sampling performed yet. Call sample() first.")
        
        summary = self.get_sampling_summary()
        
        # Calculate proportions
        total_original = sum(self.original_distribution.values())
        total_sampled = sum(self.sampled_distribution.values())
        
        # Generate markdown report
        report = [
            "# Dataset Sampling Summary",
            "",
            "## Overview",
            f"- **Random Seed**: {summary['random_seed']}",
            f"- **Sample Size**: {summary['sample_size']} records",
            f"- **Original Dataset Size**: {total_original} records",
            f"- **Sampling Method**: Stratified sampling (preserving class distribution)",
            "",
            "## Class Distribution",
            "",
            "### Original Dataset",
            ""
        ]
        
        for class_label in sorted(self.original_distribution.keys()):
            count = self.original_distribution[class_label]
            proportion = count / total_original * 100
            report.append(f"- Class {class_label}: {count} records ({proportion:.2f}%)")
        
        report.extend([
            "",
            "### Sampled Dataset",
            ""
        ])
        
        for class_label in sorted(self.sampled_distribution.keys()):
            count = self.sampled_distribution[class_label]
            proportion = count / total_sampled * 100
            report.append(f"- Class {class_label}: {count} records ({proportion:.2f}%)")
        
        report.extend([
            "",
            "## Distribution Preservation",
            ""
        ])
        
        if summary['distribution_preserved']:
            report.append("✓ **Class distribution successfully preserved** (within 5% tolerance)")
        else:
            report.append("⚠ **Class distribution deviated** (exceeds 5% tolerance)")
        
        report.extend([
            "",
            "## Reproducibility",
            "",
            f"The sampling operation is deterministic and reproducible using random seed `{summary['random_seed']}`.",
            "Running the sampling with the same seed will always produce the same results.",
            "",
            "## Notes",
            "",
            "- The original dataset is preserved and not modified.",
            f"- Total records in original dataset: {total_original}",
            f"- Total records in sampled dataset: {total_sampled}",
            "- Sampling uses stratified approach to maintain class balance.",
            ""
        ])
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Sampling report saved to {output_path}")


def sample_heart_failure_data(data_path: str, 
                              n_samples: int = 300,
                              random_seed: int = 42,
                              output_report_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to sample heart failure dataset.
    
    This is a high-level function that performs deterministic sampling on the
    heart failure dataset and optionally generates a report.
    
    Args:
        data_path (str): Path to the heart failure CSV file
        n_samples (int): Number of samples to draw. Default is 300.
        random_seed (int): Random seed for reproducibility. Default is 42.
        output_report_path (str, optional): Path to save the sampling report.
            If None, no report is generated.
    
    Returns:
        pd.DataFrame: Sampled dataset
    
    Example:
        >>> # Sample 300 records and generate report
        >>> sampled_data = sample_heart_failure_data(
        ...     'data/heart_failure.csv',
        ...     n_samples=300,
        ...     output_report_path='reports/sampling_summary.md'
        ... )
        >>> print(f"Sampled {len(sampled_data)} records")
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Create sampler
    sampler = DatasetSampler(random_seed=random_seed)
    
    # Perform sampling
    sampled_data = sampler.sample(data, n_samples=n_samples, stratify=True)
    
    # Generate report if requested
    if output_report_path:
        sampler.generate_report(output_report_path)
    
    return sampled_data
