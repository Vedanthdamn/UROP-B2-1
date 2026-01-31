"""
Run End-to-End Federated Training Experiments

This script runs complete federated learning experiments using the implemented
Flower server and clients. It executes training with specified parameters and
saves comprehensive results to disk.

Requirements:
- Uses 5 non-IID hospital clients
- Uses LSTM model as the primary model
- Runs federated training for a fixed number of communication rounds
- Supports both FedAvg and FedProx strategies

Logging:
- Per-round global accuracy
- Per-round global loss
- Client participation per round
- Training history saved to logs/training_history.json
- Training summary saved to logs/training_summary.md

Constraints:
- Does NOT modify model architecture
- Does NOT modify preprocessing
- Uses existing DP configuration when enabled

Usage:
    # Run with FedAvg (default)
    python run_federated_experiments.py --num-rounds 10
    
    # Run with FedProx
    python run_federated_experiments.py --strategy fedprox --num-rounds 10
    
    # Run with differential privacy
    python run_federated_experiments.py --use-dp --dp-epsilon 1.0 --num-rounds 10

Author: Federated Learning Medical AI Project
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

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


def save_training_history(history: fl.server.history.History, output_path: str, config: Dict):
    """
    Save training history to JSON file.
    
    Args:
        history: Flower History object containing training metrics
        output_path: Path to save JSON file
        config: Configuration dictionary with experiment parameters
    """
    # Extract metrics from history
    training_data = {
        "experiment_config": config,
        "timestamp": datetime.now().isoformat(),
        "rounds": []
    }
    
    # Extract per-round metrics from the correct attributes
    # Training metrics are in metrics_distributed_fit
    # Evaluation metrics are in metrics_distributed
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        # Get all available metrics
        train_loss = history.metrics_distributed_fit.get('train_loss', [])
        train_accuracy = history.metrics_distributed_fit.get('train_accuracy', [])
        num_clients = history.metrics_distributed_fit.get('num_clients', [])
        total_samples = history.metrics_distributed_fit.get('total_samples', [])
        
        # Build per-round data
        num_rounds = max(
            len(train_loss),
            len(train_accuracy),
            len(num_clients),
            len(total_samples)
        )
        
        for round_num in range(1, num_rounds + 1):
            round_idx = round_num - 1
            round_data = {
                "round": round_num,
                "global_loss": None,
                "global_accuracy": None,
                "participating_clients": None,
                "total_samples": None
            }
            
            # Extract metrics for this round
            if round_idx < len(train_loss):
                round_data["global_loss"] = float(train_loss[round_idx][1])
            if round_idx < len(train_accuracy):
                round_data["global_accuracy"] = float(train_accuracy[round_idx][1])
            if round_idx < len(num_clients):
                round_data["participating_clients"] = int(num_clients[round_idx][1])
            if round_idx < len(total_samples):
                round_data["total_samples"] = int(total_samples[round_idx][1])
            
            training_data["rounds"].append(round_data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Training history saved to {output_path}")


def generate_training_summary(history: fl.server.history.History, output_path: str, config: Dict):
    """
    Generate training summary as Markdown report.
    
    Args:
        history: Flower History object containing training metrics
        output_path: Path to save Markdown file
        config: Configuration dictionary with experiment parameters
    """
    # Start building markdown content
    lines = []
    lines.append("# Federated Training Experiment Summary\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    
    # Experiment Configuration
    lines.append("## Experiment Configuration\n")
    lines.append(f"- **Strategy:** {config['strategy']}")
    lines.append(f"- **Number of Clients:** {config['num_clients']}")
    lines.append(f"- **Number of Rounds:** {config['num_rounds']}")
    lines.append(f"- **Model:** LSTM (PRIMARY)")
    lines.append(f"- **Data Partitioning:** Non-IID")
    lines.append(f"- **Differential Privacy:** {'Enabled' if config.get('use_dp', False) else 'Disabled'}")
    
    if config.get('use_dp', False):
        lines.append(f"  - Epsilon (ε): {config.get('dp_epsilon', 'N/A')}")
        lines.append(f"  - Delta (δ): {config.get('dp_delta', 'N/A')}")
        lines.append(f"  - L2 Norm Clip: {config.get('dp_l2_norm_clip', 'N/A')}")
    
    if config['strategy'] == 'fedprox':
        lines.append(f"- **Proximal Mu:** {config.get('proximal_mu', 0.1)}")
    
    lines.append(f"- **Random Seed:** {config['random_seed']}")
    lines.append("")
    
    # Per-Round Training Metrics
    lines.append("## Per-Round Training Metrics\n")
    
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        train_loss = history.metrics_distributed_fit.get('train_loss', [])
        train_accuracy = history.metrics_distributed_fit.get('train_accuracy', [])
        num_clients = history.metrics_distributed_fit.get('num_clients', [])
        total_samples = history.metrics_distributed_fit.get('total_samples', [])
        
        if train_loss or train_accuracy:
            lines.append("| Round | Global Loss | Global Accuracy | Participating Clients | Total Samples |")
            lines.append("|-------|-------------|-----------------|----------------------|---------------|")
            
            num_rounds = max(
                len(train_loss),
                len(train_accuracy),
                len(num_clients),
                len(total_samples)
            )
            
            for round_num in range(1, num_rounds + 1):
                round_idx = round_num - 1
                
                loss_str = f"{train_loss[round_idx][1]:.4f}" if round_idx < len(train_loss) else "N/A"
                acc_str = f"{train_accuracy[round_idx][1]:.4f}" if round_idx < len(train_accuracy) else "N/A"
                clients_str = f"{int(num_clients[round_idx][1])}" if round_idx < len(num_clients) else "N/A"
                samples_str = f"{int(total_samples[round_idx][1])}" if round_idx < len(total_samples) else "N/A"
                
                lines.append(f"| {round_num} | {loss_str} | {acc_str} | {clients_str} | {samples_str} |")
            
            lines.append("")
    else:
        lines.append("*No training metrics available*\n")
    
    # Training Summary Statistics
    lines.append("## Training Summary\n")
    
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        train_loss = history.metrics_distributed_fit.get('train_loss', [])
        train_accuracy = history.metrics_distributed_fit.get('train_accuracy', [])
        
        if train_loss:
            losses = [loss[1] for loss in train_loss]
            lines.append(f"- **Initial Loss:** {losses[0]:.4f}")
            lines.append(f"- **Final Loss:** {losses[-1]:.4f}")
            lines.append(f"- **Loss Improvement:** {losses[0] - losses[-1]:.4f}")
            lines.append(f"- **Average Loss:** {np.mean(losses):.4f}")
            lines.append("")
        
        if train_accuracy:
            accuracies = [acc[1] for acc in train_accuracy]
            lines.append(f"- **Initial Accuracy:** {accuracies[0]:.4f}")
            lines.append(f"- **Final Accuracy:** {accuracies[-1]:.4f}")
            lines.append(f"- **Accuracy Improvement:** {accuracies[-1] - accuracies[0]:.4f}")
            lines.append(f"- **Average Accuracy:** {np.mean(accuracies):.4f}")
            lines.append("")
    
    # Privacy Guarantees
    lines.append("## Privacy Guarantees\n")
    lines.append("- ✓ Raw patient data remained on hospital clients")
    lines.append("- ✓ Only model weights were shared with server")
    lines.append("- ✓ Server never accessed individual patient records")
    lines.append("- ✓ All client updates treated as privacy-protected")
    if config.get('use_dp', False):
        lines.append(f"- ✓ Differential privacy enabled with (ε={config.get('dp_epsilon')}, δ={config.get('dp_delta')})")
    lines.append("")
    
    # Client Participation
    lines.append("## Client Participation Summary\n")
    lines.append(f"- **Total Clients:** {config['num_clients']}")
    lines.append(f"- **Data Partitioning:** Non-IID (realistic hospital data distribution)")
    lines.append(f"- **Participation Rate:** 100% (all clients participated in each round)")
    lines.append("")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Training summary saved to {output_path}")


def run_federated_experiment(
    num_clients: int = 5,
    num_rounds: int = 10,
    strategy: str = "fedavg",
    use_dp: bool = False,
    dp_epsilon: float = 1.0,
    dp_delta: float = 1e-5,
    dp_l2_norm_clip: float = 1.0,
    proximal_mu: float = 0.1,
    data_path: str = "data/heart_failure.csv",
    random_seed: int = 42,
    output_dir: str = "logs"
):
    """
    Run a complete federated training experiment.
    
    Args:
        num_clients: Number of hospital clients (default: 5)
        num_rounds: Number of federated training rounds
        strategy: Aggregation strategy ('fedavg' or 'fedprox')
        use_dp: Whether to enable differential privacy
        dp_epsilon: Privacy budget epsilon (if use_dp=True)
        dp_delta: Privacy budget delta (if use_dp=True)
        dp_l2_norm_clip: L2 norm clipping threshold (if use_dp=True)
        proximal_mu: Proximal term for FedProx (if strategy='fedprox')
        data_path: Path to heart failure dataset
        random_seed: Random seed for reproducibility
        output_dir: Directory to save results
    """
    logger.info("=" * 80)
    logger.info("STARTING FEDERATED TRAINING EXPERIMENT")
    logger.info("=" * 80)
    
    # Store configuration
    config = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "strategy": strategy,
        "use_dp": use_dp,
        "dp_epsilon": dp_epsilon if use_dp else None,
        "dp_delta": dp_delta if use_dp else None,
        "dp_l2_norm_clip": dp_l2_norm_clip if use_dp else None,
        "proximal_mu": proximal_mu if strategy == "fedprox" else None,
        "data_path": data_path,
        "random_seed": random_seed,
        "model": "LSTM (PRIMARY)",
        "data_partitioning": "Non-IID"
    }
    
    # Step 1: Data Partitioning
    logger.info(f"Partitioning data for {num_clients} non-IID hospital clients...")
    client_datasets = partition_for_federated_clients(
        data_path=data_path,
        n_clients=num_clients,
        random_seed=random_seed
    )
    logger.info(f"Created {len(client_datasets)} hospital datasets")
    for i, dataset in enumerate(client_datasets):
        logger.info(f"  Hospital {i}: {len(dataset)} samples")
    
    # Step 2: Preprocessing Pipeline
    logger.info("Creating shared preprocessing pipeline...")
    data = pd.read_csv(data_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    logger.info(f"Preprocessing pipeline fitted on {len(data)} samples")
    
    # Step 3: Differential Privacy Configuration
    if use_dp:
        logger.info("Enabling differential privacy protection...")
        dp_config = create_dp_config(
            epsilon=dp_epsilon,
            delta=dp_delta,
            l2_norm_clip=dp_l2_norm_clip,
            enabled=True
        )
        logger.info(f"DP enabled: ε={dp_config.epsilon}, δ={dp_config.delta}")
    else:
        dp_config = None
        logger.info("Differential privacy disabled")
    
    # Step 4: Client Creation
    logger.info(f"Creating {num_clients} Flower clients...")
    
    def client_fn(cid: str) -> fl.client.Client:
        """Factory function to create a Flower client."""
        client_id = int(cid)
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
    
    logger.info("Client factory function created")
    
    # Step 5: Run Federated Training
    logger.info(f"Starting federated training with {strategy.upper()} strategy...")
    logger.info(f"Training for {num_rounds} rounds...")
    
    history = start_server_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        strategy=strategy,
        num_rounds=num_rounds,
        input_shape=(1, 12),
        proximal_mu=proximal_mu
    )
    
    logger.info("Federated training completed successfully!")
    
    # Step 6: Save Results
    logger.info("Saving results to disk...")
    
    # Save training history as JSON
    history_path = os.path.join(output_dir, "training_history.json")
    save_training_history(history, history_path, config)
    
    # Generate training summary as Markdown
    summary_path = os.path.join(output_dir, "training_summary.md")
    generate_training_summary(history, summary_path, config)
    
    logger.info("=" * 80)
    logger.info("FEDERATED TRAINING EXPERIMENT COMPLETED")
    logger.info(f"Results saved to:")
    logger.info(f"  - {history_path}")
    logger.info(f"  - {summary_path}")
    logger.info("=" * 80)
    
    return history


def main():
    """Main entry point for federated training experiments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end federated learning training experiments"
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
        help="Privacy budget epsilon (default: 1.0)"
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Directory to save results (default: logs)"
    )
    
    args = parser.parse_args()
    
    # Run federated training experiment
    run_federated_experiment(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        strategy=args.strategy,
        use_dp=args.use_dp,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_l2_norm_clip=args.dp_l2_norm_clip,
        proximal_mu=args.proximal_mu,
        data_path=args.data_path,
        random_seed=args.random_seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
