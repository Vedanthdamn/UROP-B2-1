"""
Demo script for Differential Privacy in Federated Learning

This script demonstrates how to use differential privacy with Flower
federated clients for enhanced privacy-preserving training.

Usage:
    python demo_differential_privacy.py
"""

import numpy as np
import pandas as pd
from utils.client_partitioning import partition_for_federated_clients
from utils.preprocessing import create_preprocessing_pipeline
from federated import create_flower_client, create_dp_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_dp_basics():
    """Demonstrate basic DP configuration."""
    print("\n" + "=" * 80)
    print("DEMO 1: Differential Privacy Configuration")
    print("=" * 80)
    
    print("\nStep 1: Creating DP configurations with different privacy levels...")
    
    # Strong privacy (low epsilon)
    strong_privacy = create_dp_config(
        epsilon=0.5,
        delta=1e-5,
        l2_norm_clip=1.0,
        enabled=True
    )
    print(f"\n✓ Strong Privacy Config:")
    print(f"  - epsilon: {strong_privacy.epsilon} (lower = stronger privacy)")
    print(f"  - delta: {strong_privacy.delta}")
    print(f"  - l2_norm_clip: {strong_privacy.l2_norm_clip}")
    print(f"  - noise_multiplier: {strong_privacy.noise_multiplier:.4f}")
    
    # Moderate privacy (medium epsilon)
    moderate_privacy = create_dp_config(
        epsilon=1.0,
        delta=1e-5,
        l2_norm_clip=1.0,
        enabled=True
    )
    print(f"\n✓ Moderate Privacy Config:")
    print(f"  - epsilon: {moderate_privacy.epsilon}")
    print(f"  - delta: {moderate_privacy.delta}")
    print(f"  - l2_norm_clip: {moderate_privacy.l2_norm_clip}")
    print(f"  - noise_multiplier: {moderate_privacy.noise_multiplier:.4f}")
    
    # Relaxed privacy (higher epsilon)
    relaxed_privacy = create_dp_config(
        epsilon=3.0,
        delta=1e-5,
        l2_norm_clip=1.0,
        enabled=True
    )
    print(f"\n✓ Relaxed Privacy Config:")
    print(f"  - epsilon: {relaxed_privacy.epsilon}")
    print(f"  - delta: {relaxed_privacy.delta}")
    print(f"  - l2_norm_clip: {relaxed_privacy.l2_norm_clip}")
    print(f"  - noise_multiplier: {relaxed_privacy.noise_multiplier:.4f}")
    
    print("\nKey Insight:")
    print("  - Lower epsilon → more noise → stronger privacy → potentially lower utility")
    print("  - Higher epsilon → less noise → weaker privacy → potentially higher utility")
    print("  - Privacy-utility tradeoff must be carefully balanced")
    print("=" * 80)


def demo_single_client_with_dp():
    """Demonstrate training a single client with DP."""
    print("\n" + "=" * 80)
    print("DEMO 2: Single Client Training with Differential Privacy")
    print("=" * 80)
    
    # Step 1: Load and partition data
    print("\nStep 1: Loading and partitioning data for 5 hospitals...")
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=5,
        random_seed=42
    )
    print(f"✓ Created {len(client_datasets)} hospital datasets")
    
    # Step 2: Create and fit preprocessing pipeline
    print("\nStep 2: Creating shared preprocessing pipeline...")
    data = pd.read_csv('data/heart_failure.csv')
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    print("✓ Preprocessing pipeline fitted")
    
    # Step 3: Create DP configuration
    print("\nStep 3: Creating differential privacy configuration...")
    dp_config = create_dp_config(
        epsilon=1.0,
        delta=1e-5,
        l2_norm_clip=1.0,
        enabled=True
    )
    print(f"✓ DP Config: epsilon={dp_config.epsilon}, delta={dp_config.delta}")
    
    # Step 4: Create Flower client with DP
    print("\nStep 4: Creating Flower client with DP for Hospital 0...")
    client = create_flower_client(
        client_data=client_datasets[0],
        preprocessor=preprocessor,
        val_split=0.2,
        epochs_per_round=2,
        batch_size=16,
        client_id="hospital_0_with_dp",
        dp_config=dp_config
    )
    print("✓ Flower client with DP created")
    
    # Step 5: Get initial parameters
    print("\nStep 5: Getting initial model parameters...")
    params = client.get_parameters(config={})
    print(f"✓ Retrieved {len(params)} weight arrays")
    
    # Step 6: Train with DP
    print("\nStep 6: Training locally with differential privacy...")
    print(f"  Training for {client.epochs_per_round} epochs...")
    updated_params, n_samples, metrics = client.fit(params, config={})
    
    print(f"\n✓ Training completed with DP applied")
    print(f"\n  Training Metrics:")
    print(f"    - Samples used: {n_samples}")
    print(f"    - Train loss (before DP): {metrics['train_loss']:.4f}")
    print(f"    - Train loss (after DP): {metrics['train_loss_after_dp']:.4f}")
    print(f"    - Train accuracy: {metrics['train_accuracy']:.4f}")
    
    print(f"\n  Differential Privacy Metrics:")
    print(f"    - DP enabled: {metrics['dp_enabled']}")
    print(f"    - DP epsilon: {metrics['dp_epsilon']}")
    print(f"    - DP delta: {metrics['dp_delta']}")
    print(f"    - DP l2_norm_clip: {metrics['dp_l2_norm_clip']}")
    print(f"    - DP noise_multiplier: {metrics['dp_noise_multiplier']:.4f}")
    print(f"    - Original update norm: {metrics['dp_original_update_norm']:.4f}")
    print(f"    - Clipped update norm: {metrics['dp_clipped_update_norm']:.4f}")
    print(f"    - Noisy update norm: {metrics['dp_noisy_update_norm']:.4f}")
    
    print("\n✓ Privacy Guarantees:")
    print(f"  - Model updates satisfy ({dp_config.epsilon}, {dp_config.delta})-differential privacy")
    print(f"  - Raw gradients are NEVER sent to server")
    print(f"  - Only DP-protected model weights are transmitted")
    
    print("=" * 80)


def demo_comparison_with_without_dp():
    """Compare training with and without DP."""
    print("\n" + "=" * 80)
    print("DEMO 3: Comparison - Training With vs Without DP")
    print("=" * 80)
    
    # Load and partition data
    print("\nStep 1: Preparing data...")
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=3,
        random_seed=42
    )
    
    data = pd.read_csv('data/heart_failure.csv')
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    print("✓ Data prepared")
    
    # Create client WITHOUT DP
    print("\nStep 2: Training client WITHOUT differential privacy...")
    client_no_dp = create_flower_client(
        client_data=client_datasets[0],
        preprocessor=preprocessor,
        val_split=0.2,
        epochs_per_round=3,
        batch_size=16,
        client_id="hospital_no_dp",
        dp_config=None  # No DP
    )
    
    params = client_no_dp.get_parameters(config={})
    _, _, metrics_no_dp = client_no_dp.fit(params, config={})
    
    print(f"✓ Training without DP completed")
    print(f"  - Train loss: {metrics_no_dp['train_loss']:.4f}")
    print(f"  - Train accuracy: {metrics_no_dp['train_accuracy']:.4f}")
    print(f"  - Val loss: {metrics_no_dp['val_loss']:.4f}")
    print(f"  - Val accuracy: {metrics_no_dp['val_accuracy']:.4f}")
    
    # Create client WITH DP
    print("\nStep 3: Training client WITH differential privacy...")
    dp_config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
    
    client_with_dp = create_flower_client(
        client_data=client_datasets[0],
        preprocessor=preprocessor,
        val_split=0.2,
        epochs_per_round=3,
        batch_size=16,
        client_id="hospital_with_dp",
        dp_config=dp_config
    )
    
    params = client_with_dp.get_parameters(config={})
    _, _, metrics_with_dp = client_with_dp.fit(params, config={})
    
    print(f"✓ Training with DP completed")
    print(f"  - Train loss (before DP): {metrics_with_dp['train_loss']:.4f}")
    print(f"  - Train loss (after DP): {metrics_with_dp['train_loss_after_dp']:.4f}")
    print(f"  - Train accuracy: {metrics_with_dp['train_accuracy']:.4f}")
    print(f"  - Val loss: {metrics_with_dp['val_loss']:.4f}")
    print(f"  - Val accuracy: {metrics_with_dp['val_accuracy']:.4f}")
    
    # Compare results
    print("\nStep 4: Comparison Summary")
    print(f"  {'Metric':<30} {'Without DP':<15} {'With DP':<15} {'Difference':<15}")
    print(f"  {'-' * 75}")
    print(f"  {'Train Loss':<30} {metrics_no_dp['train_loss']:<15.4f} {metrics_with_dp['train_loss']:<15.4f} {metrics_with_dp['train_loss'] - metrics_no_dp['train_loss']:<15.4f}")
    print(f"  {'Train Accuracy':<30} {metrics_no_dp['train_accuracy']:<15.4f} {metrics_with_dp['train_accuracy']:<15.4f} {metrics_with_dp['train_accuracy'] - metrics_no_dp['train_accuracy']:<15.4f}")
    print(f"  {'Val Loss':<30} {metrics_no_dp['val_loss']:<15.4f} {metrics_with_dp['val_loss']:<15.4f} {metrics_with_dp['val_loss'] - metrics_no_dp['val_loss']:<15.4f}")
    print(f"  {'Val Accuracy':<30} {metrics_no_dp['val_accuracy']:<15.4f} {metrics_with_dp['val_accuracy']:<15.4f} {metrics_with_dp['val_accuracy'] - metrics_no_dp['val_accuracy']:<15.4f}")
    
    print("\n✓ Key Observations:")
    print("  - DP adds noise to model updates, which may affect training loss")
    print("  - DP provides formal privacy guarantees without exposing raw gradients")
    print("  - Privacy-utility tradeoff can be tuned via epsilon parameter")
    print("  - DP is essential for production federated learning systems")
    
    print("=" * 80)


def demo_multiple_clients_with_dp():
    """Demonstrate federated learning with multiple clients using DP."""
    print("\n" + "=" * 80)
    print("DEMO 4: Federated Learning with Differential Privacy (Multiple Clients)")
    print("=" * 80)
    
    # Step 1: Partition data
    print("\nStep 1: Partitioning data for 5 hospitals...")
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=5,
        random_seed=42
    )
    print(f"✓ Created {len(client_datasets)} hospital datasets")
    
    # Step 2: Create preprocessing pipeline
    print("\nStep 2: Creating shared preprocessing pipeline...")
    data = pd.read_csv('data/heart_failure.csv')
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    print("✓ Preprocessing pipeline fitted")
    
    # Step 3: Create DP configuration
    print("\nStep 3: Creating unified DP configuration for all clients...")
    dp_config = create_dp_config(
        epsilon=1.0,
        delta=1e-5,
        l2_norm_clip=1.0,
        enabled=True
    )
    print(f"✓ DP Config: epsilon={dp_config.epsilon}, delta={dp_config.delta}")
    
    # Step 4: Create clients with DP
    print("\nStep 4: Creating Flower clients with DP for all hospitals...")
    clients = []
    for i, hospital_data in enumerate(client_datasets):
        client = create_flower_client(
            client_data=hospital_data,
            preprocessor=preprocessor,
            val_split=0.2,
            epochs_per_round=2,
            batch_size=16,
            client_id=f"hospital_{i}_dp",
            dp_config=dp_config
        )
        clients.append(client)
        print(f"  ✓ Hospital {i}: {len(hospital_data)} samples, DP enabled")
    
    print(f"\n✓ Created {len(clients)} Flower clients with DP")
    
    # Step 5: Simulate federated training round
    print("\nStep 5: Simulating one federated training round with DP...")
    print("  (In production, this would be coordinated by Flower server)")
    
    # Get initial global weights
    global_weights = clients[0].get_parameters(config={})
    print(f"  ✓ Initial global model has {len(global_weights)} weight arrays")
    
    # Each client trains locally with DP
    client_updates = []
    print("\n  Training on each hospital with DP protection:")
    for i, client in enumerate(clients):
        updated_weights, n_samples, metrics = client.fit(global_weights, config={})
        client_updates.append({
            'weights': updated_weights,
            'n_samples': n_samples,
            'metrics': metrics
        })
        print(f"    Hospital {i}:")
        print(f"      - Samples: {n_samples}")
        print(f"      - Train loss (before DP): {metrics['train_loss']:.4f}")
        print(f"      - Train loss (after DP): {metrics['train_loss_after_dp']:.4f}")
        print(f"      - DP epsilon: {metrics['dp_epsilon']}")
        print(f"      - Update norm (clipped): {metrics['dp_clipped_update_norm']:.4f}")
    
    print("\n✓ All clients completed DP-protected local training")
    
    # Step 6: Aggregate with weighted average
    print("\nStep 6: Aggregating DP-protected updates...")
    total_samples = sum(update['n_samples'] for update in client_updates)
    
    aggregated_weights = []
    for layer_idx in range(len(global_weights)):
        layer_weights = np.zeros_like(global_weights[layer_idx])
        for update in client_updates:
            weight = update['n_samples'] / total_samples
            layer_weights += weight * update['weights'][layer_idx]
        aggregated_weights.append(layer_weights)
    
    print(f"✓ Aggregated DP-protected weights from {len(clients)} clients")
    print(f"  Total samples: {total_samples}")
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING WITH DIFFERENTIAL PRIVACY - COMPLETED")
    print("=" * 80)
    print("\nKey Privacy Features Demonstrated:")
    print("  ✓ Differential privacy applied to each client's updates")
    print("  ✓ Gradient clipping bounds sensitivity of updates")
    print("  ✓ Gaussian noise provides formal privacy guarantees")
    print("  ✓ Only DP-protected weights sent to server (NO raw gradients)")
    print("  ✓ Privacy budget (epsilon, delta) tracked and logged")
    print("  ✓ Local training loss logged after DP application")
    print("  ✓ Federated averaging combines DP-protected updates")
    print("=" * 80 + "\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("DIFFERENTIAL PRIVACY FOR FEDERATED LEARNING - DEMONSTRATIONS")
    print("=" * 80)
    print("\nThis demo shows how to use differential privacy with Flower clients")
    print("for enhanced privacy-preserving federated learning.")
    print("=" * 80)
    
    # Demo 1: DP basics
    demo_dp_basics()
    
    # Demo 2: Single client with DP
    demo_single_client_with_dp()
    
    # Demo 3: Comparison with/without DP
    demo_comparison_with_without_dp()
    
    # Demo 4: Multiple clients with DP
    demo_multiple_clients_with_dp()
    
    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Review federated/differential_privacy.py for implementation details")
    print("  2. Adjust epsilon/delta parameters based on privacy requirements")
    print("  3. Run test_differential_privacy.py to validate DP functionality")
    print("  4. Integrate DP into production federated learning workflows")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
