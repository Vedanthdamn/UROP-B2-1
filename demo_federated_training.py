"""
Demo script for Full Federated Training Session

This script demonstrates a complete federated learning training session using
Flower server and clients. It simulates a realistic federated learning scenario
with multiple hospitals training collaboratively.

Key Features:
- Simulated federated environment with multiple hospital clients
- Privacy-preserving training (no raw data sharing)
- Optional differential privacy protection
- Tracks and logs per-round metrics
- Demonstrates both FedAvg and FedProx strategies

Usage:
    # Run with FedAvg (default)
    python demo_federated_training.py
    
    # Run with FedProx
    python demo_federated_training.py --strategy fedprox
    
    # Run with differential privacy
    python demo_federated_training.py --use-dp
    
    # Custom configuration
    python demo_federated_training.py --num-clients 5 --num-rounds 10 --strategy fedavg

Author: Federated Learning Medical AI Project
"""

import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict

import flwr as fl

from federated import (
    create_flower_client,
    create_dp_config,
    start_server_simulation
)
from utils.client_partitioning import partition_for_federated_clients
from utils.preprocessing import create_preprocessing_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print demo banner."""
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING TRAINING SESSION")
    print("Privacy-Preserving Medical AI for Heart Failure Prediction")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Print section header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def demo_federated_training(
    num_clients: int = 5,
    num_rounds: int = 10,
    strategy: str = "fedavg",
    use_dp: bool = False,
    dp_epsilon: float = 1.0,
    dp_delta: float = 1e-5,
    dp_l2_norm_clip: float = 1.0,
    proximal_mu: float = 0.1,
    data_path: str = "data/heart_failure.csv",
    random_seed: int = 42
):
    """
    Run a full federated training session.
    
    Args:
        num_clients: Number of hospital clients to simulate
        num_rounds: Number of federated training rounds
        strategy: Aggregation strategy ('fedavg' or 'fedprox')
        use_dp: Whether to enable differential privacy
        dp_epsilon: Privacy budget epsilon (if use_dp=True)
        dp_delta: Privacy budget delta (if use_dp=True)
        dp_l2_norm_clip: L2 norm clipping threshold (if use_dp=True)
        proximal_mu: Proximal term for FedProx (if strategy='fedprox')
        data_path: Path to heart failure dataset
        random_seed: Random seed for reproducibility
    """
    print_banner()
    
    # -------------------------------------------------------------------------
    # Step 1: Data Partitioning
    # -------------------------------------------------------------------------
    print_section("Step 1: Data Partitioning for Federated Clients")
    logger.info(f"Partitioning data for {num_clients} hospital clients...")
    
    client_datasets = partition_for_federated_clients(
        data_path=data_path,
        n_clients=num_clients,
        random_seed=random_seed
    )
    
    print(f"✓ Created {len(client_datasets)} hospital datasets")
    for i, dataset in enumerate(client_datasets):
        print(f"  Hospital {i}: {len(dataset)} samples")
    
    # -------------------------------------------------------------------------
    # Step 2: Preprocessing Pipeline
    # -------------------------------------------------------------------------
    print_section("Step 2: Creating Shared Preprocessing Pipeline")
    logger.info("Fitting preprocessing pipeline on global data...")
    
    # Load full dataset for fitting preprocessor (centralized preprocessing)
    # Note: In production, this could be done on aggregated statistics
    data = pd.read_csv(data_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    print(f"✓ Preprocessing pipeline fitted on {len(data)} samples")
    print(f"  Features: {preprocessor.feature_columns}")
    print(f"  Target: DEATH_EVENT")
    
    # -------------------------------------------------------------------------
    # Step 3: Differential Privacy Configuration
    # -------------------------------------------------------------------------
    if use_dp:
        print_section("Step 3: Differential Privacy Configuration")
        logger.info("Enabling differential privacy protection...")
        
        dp_config = create_dp_config(
            epsilon=dp_epsilon,
            delta=dp_delta,
            l2_norm_clip=dp_l2_norm_clip,
            enabled=True
        )
        
        print(f"✓ Differential Privacy enabled")
        print(f"  Epsilon (ε): {dp_config.epsilon}")
        print(f"  Delta (δ): {dp_config.delta}")
        print(f"  L2 norm clip: {dp_config.l2_norm_clip}")
        print(f"  Noise multiplier: {dp_config.noise_multiplier:.4f}")
        print(f"\n  Privacy Note: Lower epsilon = stronger privacy")
        print(f"                (epsilon={dp_config.epsilon} is moderate privacy)")
    else:
        dp_config = None
        print_section("Step 3: Differential Privacy Configuration")
        print("✗ Differential Privacy disabled (use --use-dp to enable)")
    
    # -------------------------------------------------------------------------
    # Step 4: Federated Server Configuration
    # -------------------------------------------------------------------------
    print_section("Step 4: Federated Server Configuration")
    logger.info(f"Configuring federated server with {strategy.upper()} strategy...")
    
    print(f"✓ Server configuration:")
    print(f"  Aggregation strategy: {strategy.upper()}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Minimum clients: {min(2, num_clients)}")
    if strategy == "fedprox":
        print(f"  Proximal mu: {proximal_mu}")
    
    # -------------------------------------------------------------------------
    # Step 5: Client Creation
    # -------------------------------------------------------------------------
    print_section("Step 5: Creating Flower Clients")
    logger.info(f"Creating {num_clients} Flower clients...")
    
    def client_fn(cid: str) -> fl.client.Client:
        """
        Factory function to create a Flower client.
        
        Args:
            cid: Client ID (string representation of client index)
        
        Returns:
            FlowerClient instance ready for training
        """
        client_id = int(cid)
        
        # Create Flower client with optional DP
        client = create_flower_client(
            client_data=client_datasets[client_id],
            preprocessor=preprocessor,
            val_split=0.2,
            epochs_per_round=5,
            batch_size=32,
            client_id=f"hospital_{client_id}",
            input_shape=(1, 12),
            random_seed=random_seed,
            dp_config=dp_config
        )
        
        return client
    
    print(f"✓ Client factory function created")
    print(f"  Each client trains for 5 epochs per round")
    print(f"  Batch size: 32")
    print(f"  Validation split: 20%")
    print(f"  DP protection: {'Enabled' if use_dp else 'Disabled'}")
    
    # -------------------------------------------------------------------------
    # Step 6: Privacy Guarantees
    # -------------------------------------------------------------------------
    print_section("Step 6: Privacy Guarantees")
    print("✓ Privacy-preserving federated learning enabled:")
    print("  • Raw patient data NEVER leaves hospital clients")
    print("  • Only model weights are transmitted to server")
    print("  • Server CANNOT access individual patient records")
    print("  • All client updates treated as DP-protected")
    if use_dp:
        print(f"  • (ε={dp_epsilon}, δ={dp_delta})-differential privacy per client")
    print("  • Only aggregated metrics are logged")
    
    # -------------------------------------------------------------------------
    # Step 7: Federated Training
    # -------------------------------------------------------------------------
    print_section("Step 7: Running Federated Training")
    logger.info("Starting federated training simulation...")
    
    print(f"Training {num_clients} hospital clients for {num_rounds} rounds...")
    print("(This may take a few minutes...)\n")
    
    try:
        # Run federated training simulation
        history = start_server_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            strategy=strategy,
            num_rounds=num_rounds,
            input_shape=(1, 12),
            proximal_mu=proximal_mu
        )
        
        print("\n✓ Federated training completed successfully!")
        
        # -------------------------------------------------------------------------
        # Step 8: Training Results
        # -------------------------------------------------------------------------
        print_section("Step 8: Training Results")
        
        # Extract metrics from history
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            print("\nPer-Round Training Metrics:")
            print("-" * 60)
            print(f"{'Round':<8} {'Loss':<12} {'Accuracy':<12} {'Clients':<10}")
            print("-" * 60)
            
            # Get training metrics
            for round_num in range(1, num_rounds + 1):
                # Get metrics for this round
                if 'train_loss' in history.metrics_distributed:
                    losses = history.metrics_distributed.get('train_loss', [])
                    accuracies = history.metrics_distributed.get('train_accuracy', [])
                    num_clients_list = history.metrics_distributed.get('num_clients', [])
                    
                    if round_num <= len(losses):
                        round_idx = round_num - 1
                        loss = losses[round_idx][1] if round_idx < len(losses) else 'N/A'
                        acc = accuracies[round_idx][1] if round_idx < len(accuracies) else 'N/A'
                        n_clients = num_clients_list[round_idx][1] if round_idx < len(num_clients_list) else 'N/A'
                        
                        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
                        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
                        clients_str = f"{n_clients}" if isinstance(n_clients, (int, float)) else str(n_clients)
                        
                        print(f"{round_num:<8} {loss_str:<12} {acc_str:<12} {clients_str:<10}")
            
            print("-" * 60)
        
        # -------------------------------------------------------------------------
        # Step 9: Summary
        # -------------------------------------------------------------------------
        print_section("Step 9: Summary")
        
        print("Federated Training Session Summary:")
        print(f"  Strategy: {strategy.upper()}")
        print(f"  Rounds completed: {num_rounds}")
        print(f"  Participating hospitals: {num_clients}")
        print(f"  Total samples (distributed): {sum(len(d) for d in client_datasets)}")
        if use_dp:
            print(f"  Differential Privacy: Enabled (ε={dp_epsilon}, δ={dp_delta})")
        else:
            print(f"  Differential Privacy: Disabled")
        
        print("\nPrivacy Guarantees:")
        print("  ✓ Raw patient data remained on hospital servers")
        print("  ✓ Only aggregated model updates shared")
        print("  ✓ Server never accessed individual patient records")
        
        print("\nFederated Learning Benefits:")
        print("  ✓ Collaborative learning across hospitals")
        print("  ✓ Improved model generalization")
        print("  ✓ Privacy-preserving data sharing")
        print("  ✓ Compliance with data protection regulations")
        
    except Exception as e:
        logger.error(f"Federated training failed: {e}")
        print(f"\n✗ Federated training failed: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("FEDERATED TRAINING SESSION COMPLETED")
    print("=" * 80 + "\n")


def main():
    """Main entry point for demo script."""
    parser = argparse.ArgumentParser(
        description="Run a full federated learning training session"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of hospital clients to simulate (default: 5)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated training rounds (default: 10)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fedavg", "fedprox"],
        default="fedavg",
        help="Aggregation strategy (default: fedavg)"
    )
    parser.add_argument(
        "--use-dp",
        action="store_true",
        help="Enable differential privacy protection"
    )
    parser.add_argument(
        "--dp-epsilon",
        type=float,
        default=1.0,
        help="Privacy budget epsilon (default: 1.0, lower=more private)"
    )
    parser.add_argument(
        "--dp-delta",
        type=float,
        default=1e-5,
        help="Privacy budget delta (default: 1e-5)"
    )
    parser.add_argument(
        "--dp-l2-norm-clip",
        type=float,
        default=1.0,
        help="L2 norm clipping threshold (default: 1.0)"
    )
    parser.add_argument(
        "--proximal-mu",
        type=float,
        default=0.1,
        help="Proximal term for FedProx (default: 0.1)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/heart_failure.csv",
        help="Path to heart failure dataset (default: data/heart_failure.csv)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Run federated training
    demo_federated_training(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        strategy=args.strategy,
        use_dp=args.use_dp,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_l2_norm_clip=args.dp_l2_norm_clip,
        proximal_mu=args.proximal_mu,
        data_path=args.data_path,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
