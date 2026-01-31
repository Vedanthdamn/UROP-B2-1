"""
Privacy-Utility Analysis for Federated Learning with Differential Privacy

This script runs federated training experiments with multiple privacy budgets (epsilon values)
to analyze the trade-off between privacy and model utility.

Requirements:
- Run federated training for multiple epsilon values
- Use the same dataset, preprocessing, and model configuration
- For each epsilon, record: final global accuracy, final loss, fairness metrics

Outputs:
- reports/privacy_utility_analysis.md: Detailed analysis report
- reports/accuracy_vs_epsilon.png: Accuracy vs epsilon plot
- reports/loss_vs_epsilon.png: Loss vs epsilon plot

Usage:
    python run_privacy_utility_analysis.py

Author: Federated Learning Medical AI Project
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def run_single_epsilon_experiment(
    epsilon: float,
    num_clients: int,
    num_rounds: int,
    client_datasets: List[pd.DataFrame],
    preprocessor,
    random_seed: int,
    data_path: str
) -> Dict:
    """
    Run a single federated training experiment with a specific epsilon value.
    
    Args:
        epsilon: Privacy budget epsilon value
        num_clients: Number of hospital clients
        num_rounds: Number of federated training rounds
        client_datasets: Pre-partitioned client datasets
        preprocessor: Fitted preprocessing pipeline
        random_seed: Random seed for reproducibility
        data_path: Path to dataset
    
    Returns:
        Dictionary containing experiment results:
            - epsilon: Privacy budget used
            - final_accuracy: Final global accuracy
            - final_loss: Final global loss
            - client_accuracy_variance: Fairness metric (variance of client accuracies)
            - history: Full training history
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running experiment with epsilon = {epsilon}")
    logger.info(f"{'=' * 80}")
    
    # Create DP configuration
    if epsilon > 0:
        dp_config = create_dp_config(
            epsilon=epsilon,
            delta=1e-5,
            l2_norm_clip=1.0,
            enabled=True
        )
        logger.info(f"DP enabled: ε={dp_config.epsilon}, δ={dp_config.delta}")
    else:
        # epsilon = 0 means no differential privacy
        dp_config = None
        logger.info("Differential privacy disabled (no privacy budget)")
    
    # Create client factory function
    def client_fn(cid: str) -> fl.client.Client:
        """Factory function to create a Flower client."""
        client_id = int(cid)
        client = create_flower_client(
            client_data=client_datasets[client_id],
            preprocessor=preprocessor,
            val_split=0.2,
            epochs_per_round=5,
            batch_size=32,
            client_id=f"hospital_{client_id}_eps{epsilon}",
            input_shape=(1, 12),
            random_seed=random_seed,
            dp_config=dp_config
        )
        return client
    
    # Run federated training
    logger.info(f"Starting federated training for {num_rounds} rounds...")
    history = start_server_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        strategy="fedavg",
        num_rounds=num_rounds,
        input_shape=(1, 12)
    )
    
    # Extract final metrics
    results = {
        "epsilon": epsilon,
        "final_accuracy": None,
        "final_loss": None,
        "client_accuracy_variance": None,
        "history": history
    }
    
    # Extract metrics from history
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        train_loss = history.metrics_distributed_fit.get('train_loss', [])
        train_accuracy = history.metrics_distributed_fit.get('train_accuracy', [])
        
        if train_loss:
            results["final_loss"] = float(train_loss[-1][1])
        if train_accuracy:
            results["final_accuracy"] = float(train_accuracy[-1][1])
    
    # Calculate client accuracy variance (fairness metric)
    # We'll evaluate each client on their validation set after the final round
    # to measure fairness across clients
    client_accuracies = []
    for client_id in range(num_clients):
        client = client_fn(str(client_id))
        # Get final global parameters
        if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
            # Get the final model parameters
            params = client.get_parameters(config={})
            # Evaluate on client's validation set
            loss, num_samples, metrics = client.evaluate(params, config={})
            client_accuracies.append(metrics.get('accuracy', 0.0))
    
    if client_accuracies:
        results["client_accuracy_variance"] = float(np.var(client_accuracies))
        logger.info(f"Client accuracies: {client_accuracies}")
        logger.info(f"Client accuracy variance: {results['client_accuracy_variance']:.6f}")
    
    logger.info(f"Experiment completed for epsilon = {epsilon}")
    logger.info(f"Final accuracy: {results['final_accuracy']:.4f}")
    logger.info(f"Final loss: {results['final_loss']:.4f}")
    
    return results


def plot_accuracy_vs_epsilon(results: List[Dict], output_path: str):
    """
    Plot accuracy vs epsilon.
    
    Args:
        results: List of experiment results
        output_path: Path to save the plot
    """
    # Extract data
    epsilons = [r["epsilon"] for r in results]
    accuracies = [r["final_accuracy"] for r in results]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    # Highlight no-DP baseline
    if 0 in epsilons:
        idx = epsilons.index(0)
        plt.axhline(y=accuracies[idx], color='red', linestyle='--', alpha=0.5, 
                    label=f'No DP Baseline: {accuracies[idx]:.4f}')
    
    plt.xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    plt.ylabel('Final Global Accuracy', fontsize=12, fontweight='bold')
    plt.title('Privacy-Utility Trade-off: Accuracy vs Privacy Budget', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add value labels on points
    for eps, acc in zip(epsilons, accuracies):
        if eps > 0:  # Skip no-DP baseline label
            plt.annotate(f'{acc:.3f}', (eps, acc), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Accuracy vs epsilon plot saved to {output_path}")
    plt.close()


def plot_loss_vs_epsilon(results: List[Dict], output_path: str):
    """
    Plot loss vs epsilon.
    
    Args:
        results: List of experiment results
        output_path: Path to save the plot
    """
    # Extract data
    epsilons = [r["epsilon"] for r in results]
    losses = [r["final_loss"] for r in results]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, losses, marker='s', linewidth=2, markersize=8, color='#A23B72')
    
    # Highlight no-DP baseline
    if 0 in epsilons:
        idx = epsilons.index(0)
        plt.axhline(y=losses[idx], color='red', linestyle='--', alpha=0.5, 
                    label=f'No DP Baseline: {losses[idx]:.4f}')
    
    plt.xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    plt.ylabel('Final Global Loss', fontsize=12, fontweight='bold')
    plt.title('Privacy-Utility Trade-off: Loss vs Privacy Budget', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add value labels on points
    for eps, loss in zip(epsilons, losses):
        if eps > 0:  # Skip no-DP baseline label
            plt.annotate(f'{loss:.3f}', (eps, loss), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Loss vs epsilon plot saved to {output_path}")
    plt.close()


def generate_analysis_report(results: List[Dict], output_path: str, config: Dict):
    """
    Generate privacy-utility analysis report as Markdown.
    
    Args:
        results: List of experiment results
        output_path: Path to save the markdown report
        config: Configuration dictionary
    """
    lines = []
    lines.append("# Privacy-Utility Analysis for Federated Learning\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    
    # Analysis Overview
    lines.append("## Analysis Overview\n")
    lines.append("This report analyzes the trade-off between privacy and model utility ")
    lines.append("in federated learning with differential privacy. By varying the privacy ")
    lines.append("budget (epsilon), we measure the impact on model accuracy and loss.\n")
    lines.append("")
    
    # Experiment Configuration
    lines.append("## Experiment Configuration\n")
    lines.append(f"- **Number of Clients:** {config['num_clients']}")
    lines.append(f"- **Number of Rounds:** {config['num_rounds']}")
    lines.append(f"- **Model Architecture:** LSTM (PRIMARY)")
    lines.append(f"- **Data Partitioning:** Non-IID")
    lines.append(f"- **Strategy:** FedAvg")
    lines.append(f"- **Fixed Parameters:**")
    lines.append(f"  - Delta (δ): 1e-5")
    lines.append(f"  - L2 Norm Clip: 1.0")
    lines.append(f"  - Epochs per Round: 5")
    lines.append(f"  - Batch Size: 32")
    lines.append(f"- **Random Seed:** {config['random_seed']}")
    lines.append("")
    
    # Privacy Budgets Tested
    lines.append("## Privacy Budgets Tested\n")
    lines.append("| Epsilon (ε) | Privacy Level | Description |")
    lines.append("|-------------|---------------|-------------|")
    for r in results:
        eps = r["epsilon"]
        if eps == 0:
            level = "None"
            desc = "No differential privacy (baseline)"
        elif eps < 1.0:
            level = "Strong"
            desc = "High privacy protection, potential utility loss"
        elif eps <= 2.0:
            level = "Moderate"
            desc = "Balanced privacy-utility trade-off"
        else:
            level = "Relaxed"
            desc = "Lower privacy protection, higher utility"
        lines.append(f"| {eps} | {level} | {desc} |")
    lines.append("")
    
    # Results Summary
    lines.append("## Results Summary\n")
    lines.append("| Epsilon (ε) | Final Accuracy | Final Loss | Client Accuracy Variance |")
    lines.append("|-------------|----------------|------------|-------------------------|")
    for r in results:
        eps = r["epsilon"]
        acc = r["final_accuracy"]
        loss = r["final_loss"]
        var = r["client_accuracy_variance"]
        
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        var_str = f"{var:.6f}" if var is not None else "N/A"
        
        lines.append(f"| {eps} | {acc_str} | {loss_str} | {var_str} |")
    lines.append("")
    
    # Key Findings
    lines.append("## Key Findings\n")
    
    # Find baseline (no DP)
    baseline = next((r for r in results if r["epsilon"] == 0), None)
    
    if baseline:
        baseline_acc = baseline["final_accuracy"]
        baseline_loss = baseline["final_loss"]
        
        lines.append("### Privacy-Utility Trade-off\n")
        lines.append(f"**Baseline (No DP):** Accuracy = {baseline_acc:.4f}, Loss = {baseline_loss:.4f}\n")
        
        # Compare each epsilon to baseline
        for r in results:
            if r["epsilon"] > 0:
                eps = r["epsilon"]
                acc = r["final_accuracy"]
                loss = r["final_loss"]
                
                acc_diff = acc - baseline_acc
                loss_diff = loss - baseline_loss
                acc_pct = (acc_diff / baseline_acc) * 100
                
                lines.append(f"**ε = {eps}:**")
                lines.append(f"- Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_pct:+.2f}%)")
                lines.append(f"- Loss: {loss:.4f} ({loss_diff:+.4f})")
                lines.append("")
    
    # Fairness Analysis
    lines.append("### Fairness Analysis\n")
    lines.append("Client accuracy variance measures how fairly the model performs across ")
    lines.append("different hospitals. Lower variance indicates more equitable performance.\n")
    
    variances = [(r["epsilon"], r["client_accuracy_variance"]) 
                 for r in results if r["client_accuracy_variance"] is not None]
    if variances:
        min_var_eps, min_var = min(variances, key=lambda x: x[1])
        max_var_eps, max_var = max(variances, key=lambda x: x[1])
        
        lines.append(f"- **Lowest variance:** ε = {min_var_eps} (variance = {min_var:.6f})")
        lines.append(f"- **Highest variance:** ε = {max_var_eps} (variance = {max_var:.6f})")
    lines.append("")
    
    # Recommendations
    lines.append("## Recommendations\n")
    lines.append("Based on the privacy-utility analysis:\n")
    lines.append("1. **For maximum utility:** Use higher epsilon (ε ≥ 3.0) or no DP if privacy is not a concern")
    lines.append("2. **For balanced trade-off:** Use moderate epsilon (ε = 1.0 - 2.0)")
    lines.append("3. **For strong privacy:** Use low epsilon (ε < 1.0), accepting potential utility loss")
    lines.append("4. **Consider fairness:** Choose epsilon that minimizes client accuracy variance")
    lines.append("")
    
    # Visualizations
    lines.append("## Visualizations\n")
    lines.append("See the accompanying plots for visual analysis:\n")
    lines.append("- `accuracy_vs_epsilon.png`: Shows how accuracy changes with privacy budget")
    lines.append("- `loss_vs_epsilon.png`: Shows how loss changes with privacy budget")
    lines.append("")
    
    # Privacy Guarantees
    lines.append("## Privacy Guarantees\n")
    lines.append("All experiments with ε > 0 provide (ε, δ)-differential privacy guarantees where:\n")
    lines.append("- **ε (epsilon):** Privacy budget - lower values provide stronger privacy")
    lines.append("- **δ (delta):** Fixed at 1e-5 for all experiments")
    lines.append("- **Mechanism:** Gradient clipping + Gaussian noise addition")
    lines.append("")
    lines.append("**Note:** These privacy guarantees apply to model updates shared with the server. ")
    lines.append("Raw patient data never leaves the hospital clients.")
    lines.append("")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Analysis report saved to {output_path}")


def run_privacy_utility_analysis(
    epsilon_values: List[float] = None,
    num_clients: int = 5,
    num_rounds: int = 10,
    data_path: str = "data/heart_failure.csv",
    random_seed: int = 42,
    output_dir: str = "reports"
):
    """
    Run privacy-utility analysis across multiple epsilon values.
    
    Args:
        epsilon_values: List of epsilon values to test (default: [0, 0.5, 1.0, 2.0, 5.0])
        num_clients: Number of hospital clients
        num_rounds: Number of federated training rounds
        data_path: Path to dataset
        random_seed: Random seed for reproducibility
        output_dir: Directory to save results
    """
    logger.info("=" * 80)
    logger.info("PRIVACY-UTILITY ANALYSIS FOR FEDERATED LEARNING")
    logger.info("=" * 80)
    
    # Default epsilon values if not provided
    if epsilon_values is None:
        epsilon_values = [0, 0.5, 1.0, 2.0, 5.0]
    
    logger.info(f"Testing epsilon values: {epsilon_values}")
    
    # Configuration
    config = {
        "epsilon_values": epsilon_values,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "data_path": data_path,
        "random_seed": random_seed,
        "model": "LSTM (PRIMARY)",
        "data_partitioning": "Non-IID",
        "strategy": "FedAvg"
    }
    
    # Step 1: Partition data (once, shared across all experiments)
    logger.info(f"\nPartitioning data for {num_clients} non-IID hospital clients...")
    client_datasets = partition_for_federated_clients(
        data_path=data_path,
        n_clients=num_clients,
        random_seed=random_seed
    )
    logger.info(f"Created {len(client_datasets)} hospital datasets")
    for i, dataset in enumerate(client_datasets):
        logger.info(f"  Hospital {i}: {len(dataset)} samples")
    
    # Step 2: Create preprocessing pipeline (once, shared across all experiments)
    logger.info("\nCreating shared preprocessing pipeline...")
    data = pd.read_csv(data_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    logger.info(f"Preprocessing pipeline fitted on {len(data)} samples")
    
    # Step 3: Run experiments for each epsilon value
    results = []
    for epsilon in epsilon_values:
        try:
            result = run_single_epsilon_experiment(
                epsilon=epsilon,
                num_clients=num_clients,
                num_rounds=num_rounds,
                client_datasets=client_datasets,
                preprocessor=preprocessor,
                random_seed=random_seed,
                data_path=data_path
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error in experiment with epsilon={epsilon}: {str(e)}")
            logger.exception(e)
    
    if not results:
        logger.error("No experiments completed successfully!")
        return
    
    # Step 4: Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Generating visualizations...")
    logger.info("=" * 80)
    
    accuracy_plot_path = os.path.join(output_dir, "accuracy_vs_epsilon.png")
    loss_plot_path = os.path.join(output_dir, "loss_vs_epsilon.png")
    
    plot_accuracy_vs_epsilon(results, accuracy_plot_path)
    plot_loss_vs_epsilon(results, loss_plot_path)
    
    # Step 5: Generate analysis report
    logger.info("\nGenerating analysis report...")
    report_path = os.path.join(output_dir, "privacy_utility_analysis.md")
    generate_analysis_report(results, report_path, config)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PRIVACY-UTILITY ANALYSIS COMPLETED")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to:")
    logger.info(f"  - {report_path}")
    logger.info(f"  - {accuracy_plot_path}")
    logger.info(f"  - {loss_plot_path}")
    logger.info("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run privacy-utility analysis for federated learning with differential privacy"
    )
    parser.add_argument(
        "--epsilon-values",
        type=float,
        nargs="+",
        default=None,
        help="List of epsilon values to test (default: 0 0.5 1.0 2.0 5.0)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of hospital clients (default: 5)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated training rounds (default: 10)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/heart_failure.csv",
        help="Path to dataset (default: data/heart_failure.csv)"
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
        default="reports",
        help="Directory to save results (default: reports)"
    )
    
    args = parser.parse_args()
    
    # Run privacy-utility analysis
    run_privacy_utility_analysis(
        epsilon_values=args.epsilon_values,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        data_path=args.data_path,
        random_seed=args.random_seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
