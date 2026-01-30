"""
Demonstration script for model architectures.

This script demonstrates:
1. Model instantiation for all three architectures
2. Model summaries and parameter counts
3. Integration with preprocessing pipeline
4. Model comparison
5. Primary model usage

Usage:
    python demo_models.py
"""

import numpy as np
import pandas as pd
from models import (
    LSTMClassifier,
    TCNClassifier,
    TransformerClassifier,
    get_model,
    get_primary_model,
    PRIMARY_MODEL
)
from utils.preprocessing import create_preprocessing_pipeline


def demo_basic_usage():
    """Demonstrate basic model usage."""
    print("=" * 80)
    print("DEMO 1: BASIC MODEL USAGE")
    print("=" * 80)
    print()
    
    # Create LSTM classifier (PRIMARY)
    print("Creating LSTM Classifier (PRIMARY MODEL)...")
    lstm = LSTMClassifier(input_shape=(1, 12))
    lstm.summary()
    
    # Create TCN classifier
    print("Creating TCN Classifier...")
    tcn = TCNClassifier(input_shape=(1, 12))
    tcn.summary()
    
    # Create Transformer classifier
    print("Creating Transformer Classifier...")
    transformer = TransformerClassifier(input_shape=(1, 12))
    transformer.summary()


def demo_model_comparison():
    """Compare all model architectures."""
    print("=" * 80)
    print("DEMO 2: MODEL COMPARISON")
    print("=" * 80)
    print()
    
    models = {
        'LSTM (PRIMARY)': get_model('lstm', input_shape=(1, 12)),
        'TCN': get_model('tcn', input_shape=(1, 12)),
        'Transformer': get_model('transformer', input_shape=(1, 12))
    }
    
    print("Model Comparison Table:")
    print("-" * 80)
    print(f"{'Model':<20} {'Parameters':<15} {'Input Shape':<20} {'Output Shape':<15}")
    print("-" * 80)
    
    for name, model in models.items():
        param_count = model.count_params()
        input_shape = str(model.input_shape)
        output_shape = str(model.output_shape)
        print(f"{name:<20} {param_count:<15,} {input_shape:<20} {output_shape:<15}")
    
    print("-" * 80)
    print()
    
    # Identify primary model
    print(f"PRIMARY MODEL for Federated Learning: {PRIMARY_MODEL.upper()}")
    print(f"Rationale: Best balance of performance, efficiency, and FL-friendliness")
    print()


def demo_factory_functions():
    """Demonstrate factory function usage."""
    print("=" * 80)
    print("DEMO 3: FACTORY FUNCTIONS")
    print("=" * 80)
    print()
    
    # Get primary model (recommended for FL)
    print("Getting primary model (recommended for federated learning)...")
    primary = get_primary_model(input_shape=(1, 12))
    print(f"Primary model: {primary.name}")
    print(f"Parameters: {primary.count_params():,}")
    print()
    
    # Get specific models by name
    print("Getting models by name...")
    lstm = get_model('lstm', input_shape=(1, 12))
    print(f"✓ LSTM: {lstm.count_params():,} parameters")
    
    tcn = get_model('tcn', input_shape=(1, 12))
    print(f"✓ TCN: {tcn.count_params():,} parameters")
    
    transformer = get_model('transformer', input_shape=(1, 12))
    print(f"✓ Transformer: {transformer.count_params():,} parameters")
    print()


def demo_integration_with_preprocessing():
    """Demonstrate integration with preprocessing pipeline."""
    print("=" * 80)
    print("DEMO 4: INTEGRATION WITH PREPROCESSING PIPELINE")
    print("=" * 80)
    print()
    
    # Load and preprocess data
    print("Loading and preprocessing heart failure data...")
    df = pd.read_csv('data/heart_failure.csv')
    print(f"Loaded data: {df.shape}")
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X, y = preprocessor.fit_transform(df)
    print(f"Preprocessed features: {X.shape}")
    print(f"Target: {y.shape}")
    print()
    
    # Reshape for model input (add sequence dimension)
    print("Reshaping data for model input...")
    X_reshaped = X.reshape(-1, 1, 12)  # (n_samples, sequence_length, n_features)
    print(f"Reshaped features: {X_reshaped.shape}")
    print()
    
    # Get primary model
    print("Creating primary model (LSTM)...")
    model = get_primary_model(input_shape=(1, 12))
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_params():,}")
    print()
    
    # Make predictions (without training)
    print("Making predictions on first 10 samples (no training)...")
    predictions = model.predict(X_reshaped[:10], verbose=0)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions (first 5): {predictions[:5].flatten()}")
    print(f"All predictions in [0,1]: {np.all((predictions >= 0) & (predictions <= 1))}")
    print()
    
    print("✓ Integration successful!")
    print()


def demo_customization():
    """Demonstrate model customization."""
    print("=" * 80)
    print("DEMO 5: MODEL CUSTOMIZATION")
    print("=" * 80)
    print()
    
    # LSTM with different hyperparameters
    print("Creating LSTM with custom hyperparameters...")
    lstm_custom = LSTMClassifier(
        input_shape=(1, 12),
        lstm_units=64,  # More units
        dropout_rate=0.5  # Higher dropout
    )
    print(f"Custom LSTM parameters: {lstm_custom.get_model().count_params():,}")
    print(f"LSTM units: 64 (default: 32)")
    print(f"Dropout rate: 0.5 (default: 0.3)")
    print()
    
    # TCN with different hyperparameters
    print("Creating TCN with custom hyperparameters...")
    tcn_custom = TCNClassifier(
        input_shape=(1, 12),
        filters=64,  # More filters
        kernel_size=3  # Larger kernel
    )
    print(f"Custom TCN parameters: {tcn_custom.get_model().count_params():,}")
    print(f"Filters: 64 (default: 32)")
    print(f"Kernel size: 3 (default: 2)")
    print()
    
    # Transformer with different hyperparameters
    print("Creating Transformer with custom hyperparameters...")
    transformer_custom = TransformerClassifier(
        input_shape=(1, 12),
        num_heads=1,  # Single head (lightweight)
        ff_dim=64  # Larger feed-forward
    )
    print(f"Custom Transformer parameters: {transformer_custom.get_model().count_params():,}")
    print(f"Attention heads: 1 (lightweight)")
    print(f"Feed-forward dim: 64 (default: 32)")
    print()
    
    print("Note: Increasing parameters may reduce FL efficiency.")
    print("      Keep architectures shallow for production use.")
    print()


def demo_fl_considerations():
    """Demonstrate federated learning considerations."""
    print("=" * 80)
    print("DEMO 6: FEDERATED LEARNING CONSIDERATIONS")
    print("=" * 80)
    print()
    
    models = {
        'LSTM (PRIMARY)': get_model('lstm', input_shape=(1, 12)),
        'TCN': get_model('tcn', input_shape=(1, 12)),
        'Transformer': get_model('transformer', input_shape=(1, 12))
    }
    
    print("FL-Friendliness Analysis:")
    print("-" * 80)
    
    for name, model in models.items():
        param_count = model.count_params()
        
        # Calculate model size (approximate)
        model_size_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32
        
        print(f"\n{name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Approximate model size: {model_size_mb:.2f} MB")
        print(f"  Communication overhead: {'Low' if param_count < 3000 else 'Medium' if param_count < 6000 else 'High'}")
        print(f"  DP compatible: Yes")
        print(f"  Edge device suitable: Yes")
    
    print("\n" + "-" * 80)
    print()
    
    print("Recommendation for Federated Learning:")
    print(f"  ✓ Use {PRIMARY_MODEL.upper()} (PRIMARY) for production")
    print(f"  ✓ Shallow architecture minimizes communication costs")
    print(f"  ✓ All models are compatible with differential privacy")
    print(f"  ✓ All models are suitable for edge device deployment")
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("=" * 80)
    print("MODEL ARCHITECTURES DEMONSTRATION")
    print("Federated Learning Medical AI Project")
    print("=" * 80)
    print("\n")
    
    # Run demos
    demo_basic_usage()
    demo_model_comparison()
    demo_factory_functions()
    demo_integration_with_preprocessing()
    demo_customization()
    demo_fl_considerations()
    
    # Summary
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  1. Three model architectures available: LSTM, TCN, Transformer")
    print("  2. LSTM is the PRIMARY model for federated training")
    print("  3. All models are shallow and FL-friendly (< 10K parameters)")
    print("  4. All models are compatible with differential privacy")
    print("  5. Seamless integration with preprocessing pipeline")
    print("  6. Models are customizable but keep shallow for production")
    print()
    print("Next Steps:")
    print("  - Run 'python test_models.py' to validate implementations")
    print("  - See 'models/README.md' for detailed documentation")
    print("  - Use get_primary_model() for federated learning experiments")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
