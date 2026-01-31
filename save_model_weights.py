"""
Save Trained Model Weights

This script runs a quick federated training session and saves the model weights
for use in the inference pipeline.

Usage:
    python save_model_weights.py

Output:
    - logs/model_weights.h5: Trained model weights
    - logs/preprocessor.pkl: Fitted preprocessing pipeline

Author: Federated Learning Medical AI Project
"""

import os
import sys
import logging

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models import get_primary_model
from utils.preprocessing import create_preprocessing_pipeline
from utils.client_partitioning import partition_for_federated_clients
from federated import create_flower_client
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train and save model weights."""
    print("=" * 80)
    print("SAVE MODEL WEIGHTS")
    print("Training and saving federated model artifacts")
    print("=" * 80)
    print()
    
    # Configuration
    data_path = 'data/heart_failure.csv'
    num_clients = 5
    num_rounds = 5
    random_seed = 42
    
    # Output paths
    weights_path = 'logs/model_weights.h5'
    preprocessor_path = 'logs/preprocessor.pkl'
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    data = pd.read_csv(data_path)
    
    # Create and fit preprocessor
    logger.info("Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    # Save preprocessor
    logger.info(f"Saving preprocessor to {preprocessor_path}...")
    preprocessor.save(preprocessor_path)
    
    # Partition data for federated clients
    logger.info(f"Partitioning data for {num_clients} clients...")
    client_datasets = partition_for_federated_clients(
        data_path=data_path,
        n_clients=num_clients,
        random_seed=random_seed
    )
    
    # Create model
    logger.info("Creating model...")
    model = get_primary_model(input_shape=(1, 12))
    
    # Run federated training
    logger.info(f"Running {num_rounds} rounds of federated training...")
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"Round {round_num}/{num_rounds}...")
        
        # Get current model weights
        current_weights = model.get_weights()
        
        # Collect updates from all clients
        client_weights = []
        client_samples = []
        
        for client_id in range(num_clients):
            # Create client
            client = create_flower_client(
                client_data=client_datasets[client_id],
                preprocessor=preprocessor,
                val_split=0.2,
                epochs_per_round=5,
                batch_size=32,
                client_id=f"hospital_{client_id}"
            )
            
            # Train client (suppress output)
            updated_params, n_samples, metrics = client.fit(current_weights, config={})
            client_weights.append(updated_params)
            client_samples.append(n_samples)
        
        # Aggregate weights (FedAvg)
        total_samples = sum(client_samples)
        aggregated_weights = []
        
        for layer_idx in range(len(current_weights)):
            weighted_sum = sum(
                np.array(client_weights[i][layer_idx]) * client_samples[i]
                for i in range(num_clients)
            )
            aggregated_weights.append(weighted_sum / total_samples)
        
        # Update global model
        model.set_weights(aggregated_weights)
        
        logger.info(f"  Round {round_num} completed")
    
    # Save model weights
    logger.info(f"Saving model weights to {weights_path}...")
    model.save_weights(weights_path)
    
    print()
    print("=" * 80)
    print("MODEL ARTIFACTS SAVED:")
    print(f"  ✓ Model weights: {weights_path}")
    print(f"  ✓ Preprocessor: {preprocessor_path}")
    print()
    print("These artifacts can now be used for fast inference without retraining.")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())
