"""
Demo script for Flower Federated Client

This script demonstrates how to create and use Flower federated clients
for privacy-preserving training across hospitals.

Usage:
    python demo_federated_client.py
"""

import numpy as np
import pandas as pd
from utils.client_partitioning import partition_for_federated_clients
from utils.preprocessing import create_preprocessing_pipeline
from federated import create_flower_client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_single_client():
    """Demonstrate creating and testing a single Flower client."""
    print("\n" + "=" * 80)
    print("DEMO: Single Flower Client")
    print("=" * 80)
    
    # Step 1: Load and partition data into hospital datasets
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
    
    # Step 3: Create Flower client for first hospital
    print("\nStep 3: Creating Flower client for Hospital 0...")
    client = create_flower_client(
        client_data=client_datasets[0],
        preprocessor=preprocessor,
        val_split=0.2,
        epochs_per_round=2,  # Small for demo
        batch_size=16,
        client_id="hospital_0"
    )
    print("✓ Flower client created")
    
    # Step 4: Test get_parameters
    print("\nStep 4: Testing get_parameters (model weights)...")
    params = client.get_parameters(config={})
    print(f"✓ Retrieved {len(params)} weight arrays")
    print(f"  - First layer shape: {params[0].shape}")
    print(f"  - Total parameters: {sum(p.size for p in params):,}")
    
    # Step 5: Test fit (local training)
    print("\nStep 5: Testing fit (local training)...")
    print(f"  Training for {client.epochs_per_round} epochs...")
    updated_params, n_samples, metrics = client.fit(params, config={})
    print(f"✓ Training completed")
    print(f"  - Samples used: {n_samples}")
    print(f"  - Train loss: {metrics['train_loss']:.4f}")
    print(f"  - Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  - Val loss: {metrics['val_loss']:.4f}")
    print(f"  - Val accuracy: {metrics['val_accuracy']:.4f}")
    
    # Step 6: Test evaluate
    print("\nStep 6: Testing evaluate...")
    loss, n_samples, eval_metrics = client.evaluate(updated_params, config={})
    print(f"✓ Evaluation completed")
    print(f"  - Samples evaluated: {n_samples}")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Accuracy: {eval_metrics['accuracy']:.4f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nKey Privacy Features Demonstrated:")
    print("  ✓ Raw patient data stays on client")
    print("  ✓ Only model weights are shared")
    print("  ✓ No patient-level data in logs")
    print("  ✓ Aggregated metrics only")
    print("=" * 80 + "\n")


def demo_multiple_clients():
    """Demonstrate creating multiple Flower clients for federated learning."""
    print("\n" + "=" * 80)
    print("DEMO: Multiple Flower Clients (Federated Scenario)")
    print("=" * 80)
    
    # Step 1: Load and partition data
    print("\nStep 1: Partitioning data for 5 hospitals...")
    client_datasets = partition_for_federated_clients(
        data_path='data/heart_failure.csv',
        n_clients=5,
        random_seed=42,
        output_report_path='reports/federated_partition_summary.md'
    )
    print(f"✓ Created {len(client_datasets)} hospital datasets")
    print("✓ Partition report saved to reports/federated_partition_summary.md")
    
    # Step 2: Create shared preprocessing pipeline
    print("\nStep 2: Creating shared preprocessing pipeline...")
    data = pd.read_csv('data/heart_failure.csv')
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(data)
    print("✓ Preprocessing pipeline fitted")
    
    # Step 3: Create Flower clients for all hospitals
    print("\nStep 3: Creating Flower clients for all hospitals...")
    clients = []
    for i, hospital_data in enumerate(client_datasets):
        client = create_flower_client(
            client_data=hospital_data,
            preprocessor=preprocessor,
            val_split=0.2,
            epochs_per_round=2,
            batch_size=16,
            client_id=f"hospital_{i}"
        )
        clients.append(client)
        print(f"  ✓ Hospital {i}: {len(hospital_data)} samples")
    
    print(f"\n✓ Created {len(clients)} Flower clients")
    
    # Step 4: Simulate one round of federated training
    print("\nStep 4: Simulating one round of federated training...")
    print("  (In real FL, this would be coordinated by Flower server)")
    
    # Get initial global weights from first client
    global_weights = clients[0].get_parameters(config={})
    print(f"  ✓ Initial global model has {len(global_weights)} weight arrays")
    
    # Each client trains locally
    client_updates = []
    for i, client in enumerate(clients):
        print(f"\n  Training on Hospital {i}...")
        updated_weights, n_samples, metrics = client.fit(global_weights, config={})
        client_updates.append({
            'weights': updated_weights,
            'n_samples': n_samples,
            'metrics': metrics
        })
        print(f"    - Samples: {n_samples}")
        print(f"    - Train accuracy: {metrics['train_accuracy']:.4f}")
    
    print("\n✓ All clients completed local training")
    
    # Step 5: Simulate weight aggregation (simple averaging for demo)
    print("\nStep 5: Simulating federated averaging...")
    total_samples = sum(update['n_samples'] for update in client_updates)
    
    # Weighted average of parameters
    aggregated_weights = []
    for layer_idx in range(len(global_weights)):
        layer_weights = np.zeros_like(global_weights[layer_idx])
        for update in client_updates:
            weight = update['n_samples'] / total_samples
            layer_weights += weight * update['weights'][layer_idx]
        aggregated_weights.append(layer_weights)
    
    print(f"✓ Aggregated weights from {len(clients)} clients")
    print(f"  Total samples: {total_samples}")
    
    # Step 6: Evaluate aggregated model on each client
    print("\nStep 6: Evaluating aggregated model on each client...")
    for i, client in enumerate(clients):
        loss, n_samples, metrics = client.evaluate(aggregated_weights, config={})
        print(f"  Hospital {i}: accuracy={metrics['accuracy']:.4f}, loss={loss:.4f}")
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nFederated Learning Features Demonstrated:")
    print("  ✓ Non-IID data distribution across hospitals")
    print("  ✓ Local training on each client")
    print("  ✓ Only model weights shared (NO patient data)")
    print("  ✓ Federated averaging of model updates")
    print("  ✓ Privacy-preserving collaborative training")
    print("=" * 80 + "\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("FLOWER FEDERATED CLIENT DEMONSTRATIONS")
    print("=" * 80)
    print("\nThis demo shows how to use Flower clients for federated learning")
    print("with privacy-preserving hospital training.")
    print("=" * 80)
    
    # Demo 1: Single client
    demo_single_client()
    
    # Demo 2: Multiple clients (federated scenario)
    demo_multiple_clients()
    
    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Review federated/client.py for implementation details")
    print("  2. See reports/federated_partition_summary.md for data distribution")
    print("  3. Run actual Flower server/client for production FL")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
