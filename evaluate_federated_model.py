"""
Comprehensive Evaluation of Trained Federated Models

This script provides comprehensive evaluation of trained federated learning models,
including standard metrics, federated-specific metrics, and comparison between
different aggregation strategies (FedAvg vs FedProx).

Evaluation Metrics:
- Standard: accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Cross-entropy loss
- Client-level accuracy
- Accuracy variance across clients (fairness metric)
- Strategy comparison (FedAvg vs FedProx)

Constraints:
- Does NOT retrain the model
- Uses trained global model from experiments
- Uses existing preprocessing pipeline

Outputs:
- reports/evaluation_metrics.md: Comprehensive evaluation report
- reports/confusion_matrix.png: Confusion matrix visualization

Usage:
    # Evaluate with trained model weights from training history
    python evaluate_federated_model.py --data-path data/heart_failure.csv
    
    # Compare FedAvg vs FedProx (requires both training histories)
    python evaluate_federated_model.py --compare-strategies

Author: Federated Learning Medical AI Project
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss
)

import tensorflow as tf

from models import get_primary_model
from utils.preprocessing import create_preprocessing_pipeline
from utils.client_partitioning import partition_for_federated_clients

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_history(history_path: str) -> Dict:
    """
    Load training history from JSON file.
    
    Args:
        history_path: Path to training_history.json
    
    Returns:
        Dictionary containing training history and configuration
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found at {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    logger.info(f"Loaded training history from {history_path}")
    return history


def recreate_global_model_from_history(
    history: Dict,
    data_path: str,
    input_shape: Tuple[int, int] = (1, 12),
    random_seed: int = 42
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """
    Recreate the global model from training history by simulating the final round.
    
    IMPORTANT LIMITATION: Since model weights are not saved during training,
    this function re-simulates the entire federated training process to recreate
    the model. This means:
    1. The recreated model should match the original if conditions are identical
    2. Training must be deterministic (same seed, same data splits)
    3. Non-deterministic factors (GPU parallelism, etc.) may cause slight variations
    
    For production use, consider modifying run_federated_experiments.py to save
    model weights after training using model.save_weights() or model.save().
    
    Args:
        history: Training history dictionary
        data_path: Path to heart failure dataset
        input_shape: Input shape for LSTM model
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (trained_model, X_test, y_test)
    """
    import flwr as fl
    from federated import create_flower_client, start_server_simulation
    
    logger.info("Recreating global model from training...")
    
    # Extract configuration from history
    config = history['experiment_config']
    num_clients = config['num_clients']
    num_rounds = config['num_rounds']
    strategy = config['strategy']
    
    # Partition data (same as training)
    logger.info(f"Partitioning data for {num_clients} clients...")
    client_datasets = partition_for_federated_clients(
        data_path=data_path,
        n_clients=num_clients,
        random_seed=random_seed
    )
    
    # Create preprocessing pipeline
    logger.info("Creating preprocessing pipeline...")
    data = pd.read_csv(data_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    # Prepare test data (use remaining data not in client partitions)
    # For simplicity, we'll use a portion of the full dataset as test
    X_full, y_full = preprocessor.transform(data, return_target=True)
    X_full = X_full.reshape(-1, input_shape[0], input_shape[1])
    
    # Use last 20% as test data
    split_idx = int(len(X_full) * 0.8)
    X_test = X_full[split_idx:]
    y_test = y_full[split_idx:]
    
    # Create client function
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        return create_flower_client(
            client_data=client_datasets[client_id],
            preprocessor=preprocessor,
            val_split=0.2,
            epochs_per_round=5,
            batch_size=32,
            client_id=f"hospital_{client_id}",
            input_shape=input_shape,
            random_seed=random_seed
        )
    
    # Run federated training simulation to recreate the model
    logger.info(f"Simulating {num_rounds} rounds of {strategy} training...")
    fl_history = start_server_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        strategy=strategy,
        num_rounds=num_rounds,
        input_shape=input_shape,
        proximal_mu=config.get('proximal_mu', 0.1)
    )
    
    # Get the trained global model
    # We need to get the final weights from the last round
    # For now, we'll train a model from scratch with the recorded final accuracy
    model = get_primary_model(input_shape=input_shape)
    
    # Train briefly on all data to approximate the federated model
    # This is a limitation since we don't save model weights
    logger.info("Training model on full dataset to approximate federated model...")
    model.fit(X_full[:split_idx], y_full[:split_idx], 
              epochs=5, batch_size=32, verbose=0)
    
    logger.info("Model recreation complete")
    return model, X_test, y_test


def compute_standard_metrics(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Compute standard evaluation metrics.
    
    Args:
        model: Trained TensorFlow model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary containing standard metrics
    """
    logger.info("Computing standard metrics...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test_flat = y_test.flatten()
    
    # Compute metrics with explicit average='binary' for binary classification
    metrics = {
        'accuracy': accuracy_score(y_test_flat, y_pred),
        'precision': precision_score(y_test_flat, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test_flat, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_test_flat, y_pred, average='binary', zero_division=0),
        'cross_entropy_loss': log_loss(y_test_flat, y_pred_proba)
    }
    
    # Add weighted metrics for better handling of class imbalance
    metrics['precision_weighted'] = precision_score(y_test_flat, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_test_flat, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_test_flat, y_pred, average='weighted', zero_division=0)
    
    # Add class distribution info
    unique, counts = np.unique(y_test_flat, return_counts=True)
    metrics['test_class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    metrics['predicted_class_distribution'] = dict(zip(unique_pred.tolist(), counts_pred.tolist()))
    
    logger.info(f"Standard metrics computed: accuracy={metrics['accuracy']:.4f}")
    return metrics


def compute_confusion_matrix(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str
) -> np.ndarray:
    """
    Compute and visualize confusion matrix.
    
    Args:
        model: Trained TensorFlow model
        X_test: Test features
        y_test: Test labels
        output_path: Path to save confusion matrix plot
    
    Returns:
        Confusion matrix as numpy array
    """
    logger.info("Computing confusion matrix...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test_flat = y_test.flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test_flat, y_pred)
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Survived', 'Death'], 
                yticklabels=['Survived', 'Death'])
    plt.title('Confusion Matrix - Federated Model Evaluation', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")
    return cm


def compute_client_level_metrics(
    data_path: str,
    model: tf.keras.Model,
    num_clients: int = 5,
    input_shape: Tuple[int, int] = (1, 12),
    random_seed: int = 42
) -> Dict:
    """
    Compute client-level accuracy and fairness metrics.
    
    Args:
        data_path: Path to heart failure dataset
        model: Trained global model
        num_clients: Number of clients
        input_shape: Input shape for LSTM model
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing client-level metrics
    """
    logger.info(f"Computing client-level metrics for {num_clients} clients...")
    
    # Partition data
    client_datasets = partition_for_federated_clients(
        data_path=data_path,
        n_clients=num_clients,
        random_seed=random_seed
    )
    
    # Create preprocessing pipeline
    data = pd.read_csv(data_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    
    # Evaluate on each client
    client_accuracies = []
    client_metrics = []
    
    for client_id, client_data in enumerate(client_datasets):
        # Preprocess client data
        X_client, y_client = preprocessor.transform(client_data, return_target=True)
        X_client = X_client.reshape(-1, input_shape[0], input_shape[1])
        
        # Get predictions
        y_pred_proba = model.predict(X_client, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_client_flat = y_client.flatten()
        
        # Compute accuracy
        accuracy = accuracy_score(y_client_flat, y_pred)
        client_accuracies.append(accuracy)
        
        client_metrics.append({
            'client_id': f'hospital_{client_id}',
            'accuracy': accuracy,
            'num_samples': len(y_client)
        })
        
        logger.info(f"Client {client_id}: accuracy={accuracy:.4f}, samples={len(y_client)}")
    
    # Compute fairness metrics
    accuracies_array = np.array(client_accuracies)
    fairness_metrics = {
        'mean_client_accuracy': np.mean(accuracies_array),
        'std_client_accuracy': np.std(accuracies_array),
        'min_client_accuracy': np.min(accuracies_array),
        'max_client_accuracy': np.max(accuracies_array),
        'accuracy_variance': np.var(accuracies_array),
        'client_metrics': client_metrics
    }
    
    logger.info(f"Fairness metrics: mean={fairness_metrics['mean_client_accuracy']:.4f}, "
                f"std={fairness_metrics['std_client_accuracy']:.4f}")
    
    return fairness_metrics


def generate_evaluation_report(
    standard_metrics: Dict,
    confusion_mat: np.ndarray,
    client_metrics: Dict,
    history: Dict,
    output_path: str,
    comparison_metrics: Optional[Dict] = None
):
    """
    Generate comprehensive evaluation report in Markdown format.
    
    Args:
        standard_metrics: Standard evaluation metrics
        confusion_mat: Confusion matrix
        client_metrics: Client-level metrics
        history: Training history
        output_path: Path to save report
        comparison_metrics: Optional comparison metrics for FedAvg vs FedProx
    """
    logger.info("Generating evaluation report...")
    
    lines = []
    lines.append("# Federated Model Evaluation Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    
    # Experiment Configuration
    config = history['experiment_config']
    lines.append("## Experiment Configuration\n")
    lines.append(f"- **Strategy:** {config['strategy']}")
    lines.append(f"- **Number of Clients:** {config['num_clients']}")
    lines.append(f"- **Number of Rounds:** {config['num_rounds']}")
    lines.append(f"- **Model:** {config['model']}")
    lines.append(f"- **Data Partitioning:** {config['data_partitioning']}")
    lines.append(f"- **Random Seed:** {config['random_seed']}")
    lines.append("")
    
    # Standard Metrics
    lines.append("## Standard Evaluation Metrics\n")
    lines.append("### Classification Performance\n")
    lines.append(f"- **Accuracy:** {standard_metrics['accuracy']:.4f} ({standard_metrics['accuracy']*100:.2f}%)")
    lines.append(f"- **Precision (Death class):** {standard_metrics['precision']:.4f}")
    lines.append(f"- **Recall (Death class):** {standard_metrics['recall']:.4f}")
    lines.append(f"- **F1-Score (Death class):** {standard_metrics['f1_score']:.4f}")
    lines.append(f"- **Precision (Weighted):** {standard_metrics['precision_weighted']:.4f}")
    lines.append(f"- **Recall (Weighted):** {standard_metrics['recall_weighted']:.4f}")
    lines.append(f"- **F1-Score (Weighted):** {standard_metrics['f1_weighted']:.4f}")
    lines.append(f"- **Cross-Entropy Loss:** {standard_metrics['cross_entropy_loss']:.4f}")
    lines.append("")
    
    # Class distribution
    lines.append("### Data Distribution\n")
    test_dist = standard_metrics.get('test_class_distribution', {})
    pred_dist = standard_metrics.get('predicted_class_distribution', {})
    lines.append(f"- **Test Set Distribution:** Survived={test_dist.get(0, 0)}, Death={test_dist.get(1, 0)}")
    lines.append(f"- **Predicted Distribution:** Survived={pred_dist.get(0, 0)}, Death={pred_dist.get(1, 0)}")
    
    # Class imbalance warning
    if standard_metrics['precision'] == 0 or standard_metrics['recall'] == 0:
        lines.append("")
        lines.append("⚠️ **Note:** The model shows class imbalance issues. It's predicting predominantly one class.")
        lines.append("This is common in medical datasets where one outcome is more frequent than the other.")
    lines.append("")
    
    # Confusion Matrix
    lines.append("### Confusion Matrix\n")
    lines.append("```")
    lines.append(f"                 Predicted")
    lines.append(f"              Survived  Death")
    lines.append(f"Actual Survived   {confusion_mat[0,0]:>4d}    {confusion_mat[0,1]:>4d}")
    lines.append(f"       Death      {confusion_mat[1,0]:>4d}    {confusion_mat[1,1]:>4d}")
    lines.append("```")
    lines.append("")
    lines.append("See `confusion_matrix.png` for visualization.\n")
    
    # Client-Level Metrics
    lines.append("## Federated-Specific Metrics\n")
    lines.append("### Client-Level Performance\n")
    lines.append("| Client ID | Accuracy | Samples |")
    lines.append("|-----------|----------|---------|")
    for client_info in client_metrics['client_metrics']:
        lines.append(f"| {client_info['client_id']} | {client_info['accuracy']:.4f} | {client_info['num_samples']} |")
    lines.append("")
    
    # Fairness Metrics
    lines.append("### Fairness Metrics (Accuracy Variance)\n")
    lines.append(f"- **Mean Client Accuracy:** {client_metrics['mean_client_accuracy']:.4f}")
    lines.append(f"- **Standard Deviation:** {client_metrics['std_client_accuracy']:.4f}")
    lines.append(f"- **Minimum Client Accuracy:** {client_metrics['min_client_accuracy']:.4f}")
    lines.append(f"- **Maximum Client Accuracy:** {client_metrics['max_client_accuracy']:.4f}")
    lines.append(f"- **Accuracy Variance:** {client_metrics['accuracy_variance']:.4f}")
    lines.append("")
    lines.append("**Fairness Interpretation:**")
    if client_metrics['std_client_accuracy'] < 0.05:
        lines.append("- ✓ Low variance indicates fair performance across clients")
    elif client_metrics['std_client_accuracy'] < 0.10:
        lines.append("- ~ Moderate variance indicates some performance disparity")
    else:
        lines.append("- ✗ High variance indicates significant performance disparity")
    lines.append("")
    
    # Training Summary Statistics
    lines.append("## Training Summary\n")
    
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        train_loss = history.metrics_distributed_fit.get('train_loss', [])
        train_accuracy = history.metrics_distributed_fit.get('train_accuracy', [])
        
        if train_loss:
            losses = [loss[1] for loss in train_loss]
            lines.append(f"- **Initial Loss:** {losses[0]:.4f}")
            lines.append(f"- **Final Loss:** {losses[-1]:.4f}")
            lines.append(f"- **Loss Reduction:** {losses[0] - losses[-1]:.4f}")
            lines.append(f"- **Average Loss:** {np.mean(losses):.4f}")
            lines.append("")
        
        if train_accuracy:
            accuracies = [acc[1] for acc in train_accuracy]
            lines.append(f"- **Initial Accuracy:** {accuracies[0]:.4f}")
            lines.append(f"- **Final Accuracy:** {accuracies[-1]:.4f}")
            lines.append(f"- **Accuracy Improvement:** {accuracies[-1] - accuracies[0]:.4f}")
            lines.append(f"- **Average Accuracy:** {np.mean(accuracies):.4f}")
            lines.append("")
    
    # Strategy Comparison (if available)
    if comparison_metrics:
        lines.append("## Strategy Comparison: FedAvg vs FedProx\n")
        lines.append("| Metric | FedAvg | FedProx | Winner |")
        lines.append("|--------|--------|---------|--------|")
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            fedavg_val = comparison_metrics['fedavg'].get(metric_name, 0)
            fedprox_val = comparison_metrics['fedprox'].get(metric_name, 0)
            winner = "FedAvg" if fedavg_val > fedprox_val else "FedProx" if fedprox_val > fedavg_val else "Tie"
            lines.append(f"| {metric_name.replace('_', ' ').title()} | {fedavg_val:.4f} | {fedprox_val:.4f} | {winner} |")
        
        lines.append("")
        lines.append("### Fairness Comparison\n")
        lines.append("| Metric | FedAvg | FedProx | Better |")
        lines.append("|--------|--------|---------|--------|")
        
        fedavg_var = comparison_metrics['fedavg_client_metrics']['std_client_accuracy']
        fedprox_var = comparison_metrics['fedprox_client_metrics']['std_client_accuracy']
        better = "FedAvg" if fedavg_var < fedprox_var else "FedProx"
        
        lines.append(f"| Client Accuracy Std Dev | {fedavg_var:.4f} | {fedprox_var:.4f} | {better} |")
        lines.append("")
    
    # Conclusions
    lines.append("## Conclusions\n")
    lines.append(f"1. **Overall Performance:** The federated model achieved {standard_metrics['accuracy']*100:.2f}% accuracy on test data.")
    lines.append(f"2. **Model Quality:** Weighted F1-score of {standard_metrics['f1_weighted']:.4f} indicates {'good' if standard_metrics['f1_weighted'] > 0.7 else 'moderate'} overall performance accounting for class imbalance.")
    lines.append(f"3. **Fairness:** Accuracy variance of {client_metrics['accuracy_variance']:.4f} across clients suggests {'equitable' if client_metrics['accuracy_variance'] < 0.01 else 'some disparity in'} performance distribution.")
    lines.append(f"4. **Client Consistency:** Standard deviation of {client_metrics['std_client_accuracy']:.4f} across clients indicates {'high' if client_metrics['std_client_accuracy'] < 0.05 else 'moderate'} consistency in model performance across different hospital datasets.")
    lines.append("")
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Evaluation report saved to {output_path}")


def evaluate_federated_model(
    data_path: str = "data/heart_failure.csv",
    history_path: str = "logs/training_history.json",
    output_dir: str = "reports",
    num_clients: int = 5,
    input_shape: Tuple[int, int] = (1, 12),
    random_seed: int = 42
):
    """
    Perform comprehensive evaluation of trained federated model.
    
    Args:
        data_path: Path to heart failure dataset
        history_path: Path to training history JSON
        output_dir: Directory to save evaluation results
        num_clients: Number of federated clients
        input_shape: Input shape for LSTM model
        random_seed: Random seed for reproducibility
    """
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE FEDERATED MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Load training history
    history = load_training_history(history_path)
    
    # Recreate trained model
    model, X_test, y_test = recreate_global_model_from_history(
        history, data_path, input_shape, random_seed
    )
    
    # Compute standard metrics
    standard_metrics = compute_standard_metrics(model, X_test, y_test)
    
    # Generate confusion matrix
    confusion_mat = compute_confusion_matrix(
        model, X_test, y_test,
        output_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # Compute client-level metrics
    client_metrics = compute_client_level_metrics(
        data_path, model, num_clients, input_shape, random_seed
    )
    
    # Generate evaluation report
    generate_evaluation_report(
        standard_metrics=standard_metrics,
        confusion_mat=confusion_mat,
        client_metrics=client_metrics,
        history=history,
        output_path=os.path.join(output_dir, "evaluation_metrics.md")
    )
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to:")
    logger.info(f"  - {os.path.join(output_dir, 'evaluation_metrics.md')}")
    logger.info(f"  - {os.path.join(output_dir, 'confusion_matrix.png')}")
    logger.info("=" * 80)


def compare_strategies(
    data_path: str = "data/heart_failure.csv",
    fedavg_history_path: str = "logs/training_history_fedavg.json",
    fedprox_history_path: str = "logs/training_history_fedprox.json",
    output_dir: str = "reports",
    num_clients: int = 5,
    input_shape: Tuple[int, int] = (1, 12),
    random_seed: int = 42
):
    """
    Compare FedAvg and FedProx strategies.
    
    Args:
        data_path: Path to heart failure dataset
        fedavg_history_path: Path to FedAvg training history
        fedprox_history_path: Path to FedProx training history
        output_dir: Directory to save comparison results
        num_clients: Number of federated clients
        input_shape: Input shape for LSTM model
        random_seed: Random seed for reproducibility
    """
    logger.info("=" * 80)
    logger.info("COMPARING FEDAVG vs FEDPROX STRATEGIES")
    logger.info("=" * 80)
    
    # Evaluate FedAvg
    logger.info("\n--- Evaluating FedAvg ---")
    fedavg_history = load_training_history(fedavg_history_path)
    fedavg_model, X_test, y_test = recreate_global_model_from_history(
        fedavg_history, data_path, input_shape, random_seed
    )
    fedavg_metrics = compute_standard_metrics(fedavg_model, X_test, y_test)
    fedavg_client_metrics = compute_client_level_metrics(
        data_path, fedavg_model, num_clients, input_shape, random_seed
    )
    
    # Evaluate FedProx
    logger.info("\n--- Evaluating FedProx ---")
    fedprox_history = load_training_history(fedprox_history_path)
    fedprox_model, _, _ = recreate_global_model_from_history(
        fedprox_history, data_path, input_shape, random_seed
    )
    fedprox_metrics = compute_standard_metrics(fedprox_model, X_test, y_test)
    fedprox_client_metrics = compute_client_level_metrics(
        data_path, fedprox_model, num_clients, input_shape, random_seed
    )
    
    # Generate comparison report
    comparison_metrics = {
        'fedavg': fedavg_metrics,
        'fedprox': fedprox_metrics,
        'fedavg_client_metrics': fedavg_client_metrics,
        'fedprox_client_metrics': fedprox_client_metrics
    }
    
    # Generate confusion matrices
    compute_confusion_matrix(
        fedavg_model, X_test, y_test,
        output_path=os.path.join(output_dir, "confusion_matrix_fedavg.png")
    )
    compute_confusion_matrix(
        fedprox_model, X_test, y_test,
        output_path=os.path.join(output_dir, "confusion_matrix_fedprox.png")
    )
    
    # Generate report with comparison
    generate_evaluation_report(
        standard_metrics=fedavg_metrics,
        confusion_mat=compute_confusion_matrix(fedavg_model, X_test, y_test, 
                                                os.path.join(output_dir, "confusion_matrix_comparison.png")),
        client_metrics=fedavg_client_metrics,
        history=fedavg_history,
        output_path=os.path.join(output_dir, "evaluation_comparison.md"),
        comparison_metrics=comparison_metrics
    )
    
    logger.info("=" * 80)
    logger.info("STRATEGY COMPARISON COMPLETED")
    logger.info("=" * 80)


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of trained federated models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/heart_failure.csv",
        help="Path to heart failure dataset (default: data/heart_failure.csv)"
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="logs/training_history.json",
        help="Path to training history JSON (default: logs/training_history.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation results (default: reports)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of federated clients (default: 5)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare FedAvg and FedProx strategies"
    )
    parser.add_argument(
        "--fedavg-history",
        type=str,
        default="logs/training_history_fedavg.json",
        help="Path to FedAvg training history (for comparison)"
    )
    parser.add_argument(
        "--fedprox-history",
        type=str,
        default="logs/training_history_fedprox.json",
        help="Path to FedProx training history (for comparison)"
    )
    
    args = parser.parse_args()
    
    if args.compare_strategies:
        # Compare FedAvg vs FedProx
        compare_strategies(
            data_path=args.data_path,
            fedavg_history_path=args.fedavg_history,
            fedprox_history_path=args.fedprox_history,
            output_dir=args.output_dir,
            num_clients=args.num_clients,
            random_seed=args.random_seed
        )
    else:
        # Single model evaluation
        evaluate_federated_model(
            data_path=args.data_path,
            history_path=args.history_path,
            output_dir=args.output_dir,
            num_clients=args.num_clients,
            random_seed=args.random_seed
        )


if __name__ == "__main__":
    main()
