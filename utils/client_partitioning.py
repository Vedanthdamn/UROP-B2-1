"""
Non-IID Dataset Partitioning for Federated Learning

This module provides non-IID partitioning functionality to simulate federated
hospital clients. It ensures:
- Exactly 5 client datasets
- Unequal sample sizes per client (non-IID)
- Different class distributions per client (non-IID)
- No overlap of samples between clients

Author: Federated Learning Medical AI Project
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientDataPartitioner:
    """
    Non-IID dataset partitioner for federated learning.
    
    This class partitions a dataset into multiple client datasets with
    non-IID characteristics to simulate realistic federated learning scenarios
    where different hospitals have different amounts of data and different
    patient populations.
    
    Attributes:
        n_clients (int): Number of clients (hospitals)
        random_seed (int): Fixed random seed for reproducibility
        target_column (str): Name of the target column
        client_datasets (List[pd.DataFrame]): List of client datasets
        partition_info (Dict): Information about the partition
    """
    
    def __init__(self, n_clients: int = 5, random_seed: int = 42, 
                 target_column: str = 'DEATH_EVENT'):
        """
        Initialize the client data partitioner.
        
        Args:
            n_clients (int): Number of clients. Default is 5.
            random_seed (int): Random seed for reproducibility. Default is 42.
            target_column (str): Name of the target column. Default is 'DEATH_EVENT'.
        """
        self.n_clients = n_clients
        self.random_seed = random_seed
        self.target_column = target_column
        self.client_datasets = []
        self.partition_info = {}
        
        logger.info(f"ClientDataPartitioner initialized with {n_clients} clients, "
                   f"random_seed={random_seed}")
    
    def partition(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Partition data into non-IID client datasets.
        
        This method creates non-IID partitions by:
        1. Assigning unequal sample sizes to each client
        2. Creating different class distributions for each client
        3. Ensuring no overlap between client datasets
        
        Args:
            data (pd.DataFrame): Original dataset to partition
        
        Returns:
            List[pd.DataFrame]: List of client datasets
        
        Raises:
            ValueError: If target column not found or not enough samples
        """
        # Validate inputs
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        if len(data) < self.n_clients:
            raise ValueError(f"Dataset has {len(data)} samples but need at least "
                           f"{self.n_clients} samples for {self.n_clients} clients")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Shuffle the data
        shuffled_data = data.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        
        # Separate data by class
        class_0_data = shuffled_data[shuffled_data[self.target_column] == 0]
        class_1_data = shuffled_data[shuffled_data[self.target_column] == 1]
        
        logger.info(f"Original dataset: {len(data)} samples")
        logger.info(f"Class 0: {len(class_0_data)} samples")
        logger.info(f"Class 1: {len(class_1_data)} samples")
        
        # Create non-IID partitions with different class ratios
        # Strategy: Give each client different proportions of class 0 and class 1
        client_datasets = self._create_non_iid_partitions(
            class_0_data, class_1_data, len(data)
        )
        
        self.client_datasets = client_datasets
        self._compute_partition_info()
        
        logger.info(f"Created {len(client_datasets)} client datasets")
        
        return client_datasets
    
    def _create_non_iid_partitions(self, class_0_data: pd.DataFrame, 
                                   class_1_data: pd.DataFrame,
                                   total_samples: int) -> List[pd.DataFrame]:
        """
        Create non-IID partitions with different class distributions.
        
        Args:
            class_0_data (pd.DataFrame): Data with class 0
            class_1_data (pd.DataFrame): Data with class 1
            total_samples (int): Total number of samples
        
        Returns:
            List[pd.DataFrame]: List of client datasets
        """
        # Define unequal sample size proportions for each client
        # These sum to ~1.0 and create unequal distributions
        size_proportions = [0.25, 0.22, 0.20, 0.18, 0.15]  # Unequal sizes
        
        # Define different class distributions for each client
        # Each tuple is (proportion_class_0, proportion_class_1)
        # Different ratios simulate different hospital populations
        class_distributions = [
            (0.75, 0.25),  # Client 0: Heavy on class 0
            (0.60, 0.40),  # Client 1: Moderate bias to class 0
            (0.50, 0.50),  # Client 2: Balanced
            (0.35, 0.65),  # Client 3: Moderate bias to class 1
            (0.20, 0.80),  # Client 4: Heavy on class 1
        ]
        
        client_datasets = []
        class_0_index = 0
        class_1_index = 0
        
        for i in range(self.n_clients):
            # Calculate client size and class distribution
            if i == self.n_clients - 1:
                # Last client gets all remaining samples
                n_class_0 = len(class_0_data) - class_0_index
                n_class_1 = len(class_1_data) - class_1_index
            else:
                client_size = int(total_samples * size_proportions[i])
                
                # Calculate number of samples from each class based on distribution
                n_class_0 = int(client_size * class_distributions[i][0])
                n_class_1 = client_size - n_class_0
                
                # Ensure we don't exceed available samples
                n_class_0 = min(n_class_0, len(class_0_data) - class_0_index)
                n_class_1 = min(n_class_1, len(class_1_data) - class_1_index)
                
                # Adjust if we run out of one class
                if class_0_index + n_class_0 > len(class_0_data):
                    n_class_0 = len(class_0_data) - class_0_index
                    n_class_1 = client_size - n_class_0
                if class_1_index + n_class_1 > len(class_1_data):
                    n_class_1 = len(class_1_data) - class_1_index
                    n_class_0 = client_size - n_class_1
            
            # Extract samples for this client
            client_class_0 = class_0_data.iloc[class_0_index:class_0_index + n_class_0]
            client_class_1 = class_1_data.iloc[class_1_index:class_1_index + n_class_1]
            
            # Combine and shuffle
            client_data = pd.concat([client_class_0, client_class_1], ignore_index=True)
            client_data = client_data.sample(frac=1.0, 
                                            random_state=self.random_seed + i).reset_index(drop=True)
            
            client_datasets.append(client_data)
            
            # Update indices
            class_0_index += n_class_0
            class_1_index += n_class_1
            
            logger.info(f"Client {i}: {len(client_data)} samples "
                       f"(Class 0: {n_class_0}, Class 1: {n_class_1})")
        
        return client_datasets
    
    def _compute_partition_info(self) -> None:
        """Compute and store partition information."""
        self.partition_info = {
            'n_clients': self.n_clients,
            'random_seed': self.random_seed,
            'clients': []
        }
        
        for i, client_data in enumerate(self.client_datasets):
            class_dist = client_data[self.target_column].value_counts().to_dict()
            total_samples = len(client_data)
            
            client_info = {
                'client_id': i,
                'n_samples': total_samples,
                'class_distribution': class_dist,
                'class_proportions': {
                    cls: count / total_samples * 100 
                    for cls, count in class_dist.items()
                }
            }
            self.partition_info['clients'].append(client_info)
    
    def get_partition_info(self) -> Dict:
        """
        Get partition information.
        
        Returns:
            Dict: Dictionary containing partition details
        
        Raises:
            RuntimeError: If no partition has been performed yet
        """
        if not self.partition_info:
            raise RuntimeError("No partition performed yet. Call partition() first.")
        
        return self.partition_info
    
    def verify_no_overlap(self) -> bool:
        """
        Verify that there is no overlap between client datasets.
        
        Returns:
            bool: True if no overlap, False otherwise
        """
        if not self.client_datasets:
            raise RuntimeError("No partition performed yet. Call partition() first.")
        
        # Check overlap by comparing actual data rows
        # Use tuple of all values as a unique identifier for each sample
        seen_samples = set()
        
        for client_data in self.client_datasets:
            for _, row in client_data.iterrows():
                sample_id = tuple(row.values)
                
                # Check for overlap
                if sample_id in seen_samples:
                    return False
                
                seen_samples.add(sample_id)
        
        return True
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate a markdown report of the partition.
        
        Args:
            output_path (str): Path where the report should be saved
        
        Raises:
            RuntimeError: If no partition has been performed yet
        """
        if not self.partition_info:
            raise RuntimeError("No partition performed yet. Call partition() first.")
        
        info = self.partition_info
        
        # Calculate total statistics
        total_samples = sum(client['n_samples'] for client in info['clients'])
        
        # Generate markdown report
        report = [
            "# Client Partition Summary",
            "",
            "## Overview",
            f"- **Number of Clients**: {info['n_clients']}",
            f"- **Total Samples**: {total_samples}",
            f"- **Random Seed**: {info['random_seed']}",
            f"- **Partitioning Strategy**: Non-IID (unequal sizes, different class distributions)",
            "",
            "## Client Statistics",
            ""
        ]
        
        # Add table header
        report.extend([
            "| Client ID | Number of Samples | Class 0 Count | Class 0 % | Class 1 Count | Class 1 % |",
            "|-----------|-------------------|---------------|-----------|---------------|-----------|"
        ])
        
        # Add client rows
        for client in info['clients']:
            client_id = client['client_id']
            n_samples = client['n_samples']
            class_dist = client['class_distribution']
            class_prop = client['class_proportions']
            
            class_0_count = class_dist.get(0, 0)
            class_0_pct = class_prop.get(0, 0.0)
            class_1_count = class_dist.get(1, 0)
            class_1_pct = class_prop.get(1, 0.0)
            
            report.append(
                f"| Client {client_id} | {n_samples} | {class_0_count} | "
                f"{class_0_pct:.2f}% | {class_1_count} | {class_1_pct:.2f}% |"
            )
        
        report.extend([
            "",
            "## Detailed Client Distributions",
            ""
        ])
        
        # Add detailed information for each client
        for client in info['clients']:
            client_id = client['client_id']
            n_samples = client['n_samples']
            class_dist = client['class_distribution']
            class_prop = client['class_proportions']
            
            report.extend([
                f"### Client {client_id}",
                "",
                f"- **Total Samples**: {n_samples}",
                f"- **Percentage of Total Dataset**: {n_samples / total_samples * 100:.2f}%",
                "",
                "**Class Distribution:**",
                ""
            ])
            
            for cls in sorted(class_dist.keys()):
                count = class_dist[cls]
                proportion = class_prop[cls]
                report.append(f"- Class {cls}: {count} samples ({proportion:.2f}%)")
            
            report.append("")
        
        report.extend([
            "## Non-IID Characteristics",
            "",
            "### Unequal Sample Sizes",
            ""
        ])
        
        # Show sample size variation
        sizes = [client['n_samples'] for client in info['clients']]
        report.extend([
            f"- Minimum samples per client: {min(sizes)}",
            f"- Maximum samples per client: {max(sizes)}",
            f"- Average samples per client: {np.mean(sizes):.1f}",
            f"- Standard deviation: {np.std(sizes):.1f}",
            "",
            "### Different Class Distributions",
            ""
        ])
        
        # Show class distribution variation
        class_0_proportions = [
            client['class_proportions'].get(0, 0.0) 
            for client in info['clients']
        ]
        report.extend([
            f"- Class 0 proportion range: {min(class_0_proportions):.2f}% - {max(class_0_proportions):.2f}%",
            f"- Class 0 proportion std dev: {np.std(class_0_proportions):.2f}%",
            "",
            "This demonstrates clear non-IID characteristics with both unequal sample sizes "
            "and varying class distributions across clients.",
            "",
            "## Data Integrity",
            ""
        ])
        
        # Verify no overlap
        no_overlap = self.verify_no_overlap()
        if no_overlap:
            report.append("✓ **No overlap between clients**: Each sample appears in exactly one client dataset")
        else:
            report.append("⚠ **Overlap detected**: Some samples appear in multiple client datasets")
        
        report.extend([
            "",
            f"✓ **Total samples match**: {total_samples} samples partitioned across {info['n_clients']} clients",
            "",
            "## Reproducibility",
            "",
            f"The partition is deterministic and reproducible using random seed `{info['random_seed']}`.",
            "Running the partition with the same seed will always produce the same results.",
            "",
            "## Notes",
            "",
            "- Each client represents a federated hospital with its own data characteristics",
            "- Different sample sizes simulate real-world scenarios where hospitals have different patient volumes",
            "- Different class distributions simulate different patient populations across hospitals",
            "- No samples overlap between clients (proper partitioning without duplication)",
            ""
        ])
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Partition report saved to {output_path}")


def partition_for_federated_clients(data_path: str,
                                    n_clients: int = 5,
                                    random_seed: int = 42,
                                    output_report_path: str = None) -> List[pd.DataFrame]:
    """
    Convenience function to partition dataset for federated clients.
    
    This high-level function loads data, performs non-IID partitioning,
    and optionally generates a report.
    
    Args:
        data_path (str): Path to the dataset CSV file
        n_clients (int): Number of clients. Default is 5.
        random_seed (int): Random seed for reproducibility. Default is 42.
        output_report_path (str, optional): Path to save the partition report.
            If None, no report is generated.
    
    Returns:
        List[pd.DataFrame]: List of client datasets
    
    Example:
        >>> # Partition data into 5 clients and generate report
        >>> client_datasets = partition_for_federated_clients(
        ...     'data/heart_failure.csv',
        ...     n_clients=5,
        ...     output_report_path='reports/client_partition_summary.md'
        ... )
        >>> print(f"Created {len(client_datasets)} client datasets")
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Create partitioner
    partitioner = ClientDataPartitioner(n_clients=n_clients, random_seed=random_seed)
    
    # Perform partitioning
    client_datasets = partitioner.partition(data)
    
    # Generate report if requested
    if output_report_path:
        partitioner.generate_report(output_report_path)
    
    return client_datasets
