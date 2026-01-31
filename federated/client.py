"""
Flower Federated Client for Local Hospital Training

This module implements Flower federated learning clients for privacy-preserving
training across multiple hospitals. Each client trains locally on its own
non-IID hospital dataset without sharing raw patient data.

Key Features:
- Uses TensorFlow LSTM model (PRIMARY model)
- Loads non-IID hospital dataset per client
- Applies shared preprocessing pipeline
- Trains locally for fixed epochs per round
- Returns only model weights (NO patient data)
- Privacy safeguards: No patient-level data logging
- Configurable and reusable client implementation

Design Principles:
- Raw patient data NEVER leaves the client
- Only model weights are shared with the server
- No logging of patient-level data
- Clients are stateless between rounds
- Compatible with Flower's federated averaging (FedAvg)

Author: Federated Learning Medical AI Project
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
)

from models import get_primary_model
from utils.preprocessing import HeartFailurePreprocessor
from .differential_privacy import DifferentialPrivacy, DPConfig

# Configure logging (no patient data logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlowerClient(fl.client.NumPyClient):
    """
    Flower federated learning client for hospital training.
    
    This client implements the NumPyClient interface from Flower,
    enabling federated training across hospitals while preserving
    patient privacy.
    
    Key Privacy Features:
    - Raw patient data remains on the client
    - Only model weights are sent to the server
    - No patient-level data is logged
    - Aggregated metrics only (no individual predictions)
    - Optional Differential Privacy (DP) protection
    
    Attributes:
        model (tf.keras.Model): TensorFlow LSTM model for training
        X_train (np.ndarray): Preprocessed training features (local only)
        y_train (np.ndarray): Training labels (local only)
        X_val (np.ndarray): Preprocessed validation features (local only)
        y_val (np.ndarray): Validation labels (local only)
        epochs_per_round (int): Number of local training epochs per round
        batch_size (int): Batch size for training
        client_id (str): Client identifier (for logging only)
        dp_config (DPConfig): Differential privacy configuration (optional)
        dp_mechanism (DifferentialPrivacy): DP mechanism instance (if enabled)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs_per_round: int = 5,
        batch_size: int = 32,
        client_id: str = "unknown",
        dp_config: Optional[DPConfig] = None
    ):
        """
        Initialize the Flower client.
        
        Args:
            model: TensorFlow LSTM model (PRIMARY model)
            X_train: Preprocessed training features (stays local)
            y_train: Training labels (stays local)
            X_val: Optional validation features (stays local)
            y_val: Optional validation labels (stays local)
            epochs_per_round: Number of local training epochs per round
            batch_size: Batch size for training
            client_id: Client identifier for logging
            dp_config: Optional differential privacy configuration
        
        Privacy Note:
            All data (X_train, y_train, X_val, y_val) remains on the client
            and is NEVER sent to the server. Only model weights are shared.
            If dp_config is provided, differential privacy is applied to
            model updates before they are sent to the server.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.client_id = client_id
        self.dp_config = dp_config
        
        # Initialize DP mechanism if config provided
        if self.dp_config is not None and self.dp_config.enabled:
            self.dp_mechanism = DifferentialPrivacy(self.dp_config)
        else:
            self.dp_mechanism = None
        
        # Log client initialization (NO patient data)
        logger.info(
            f"Client {self.client_id} initialized: "
            f"train_samples={len(X_train)}, "
            f"val_samples={len(X_val) if X_val is not None else 0}, "
            f"epochs_per_round={epochs_per_round}, "
            f"dp_enabled={self.dp_mechanism is not None}"
        )
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Get model parameters (weights) to send to the server.
        
        This returns only the model weights, ensuring no patient data
        is transmitted.
        
        Args:
            config: Configuration dictionary from server
        
        Returns:
            List of numpy arrays containing model weights
        
        Privacy Note:
            Only returns model weights - NO patient data.
        """
        return self.model.get_weights()
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters (weights) received from the server.
        
        Args:
            parameters: List of numpy arrays containing model weights
        """
        self.model.set_weights(parameters)
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model locally on client's hospital data.
        
        This method:
        1. Receives global model weights from server
        2. Trains locally on hospital data for fixed epochs
        3. Applies differential privacy (if enabled) to model updates
        4. Returns DP-protected weights (NO patient data)
        
        Args:
            parameters: Global model weights from server
            config: Training configuration from server
        
        Returns:
            Tuple containing:
                - Updated model weights (NDArrays) - DP-protected if enabled
                - Number of training samples (int)
                - Training metrics (Dict)
        
        Privacy Note:
            - Trains on local data only
            - If DP is enabled, applies gradient clipping and noise BEFORE sending weights
            - Returns only model weights and aggregated metrics
            - NO individual patient data or predictions returned
        """
        # Set global model weights
        self.set_parameters(parameters)
        
        # Store original weights for DP computation
        original_weights = self.model.get_weights()
        
        # Log training start (NO patient data)
        logger.info(
            f"Client {self.client_id} starting local training: "
            f"samples={len(self.X_train)}, epochs={self.epochs_per_round}"
        )
        
        # Train model locally
        # Note: verbose=0 prevents logging of individual batch metrics
        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs_per_round,
            validation_data=(self.X_val, self.y_val) if self.X_val is not None else None,
            verbose=0  # Suppress individual batch output (privacy)
        )
        
        # Get updated model weights (before DP)
        updated_weights = self.model.get_weights()
        
        # Apply differential privacy if enabled
        if self.dp_mechanism is not None:
            # Apply DP to weight updates BEFORE sending to server
            dp_weights, dp_metrics = self.dp_mechanism.apply_dp_to_updates(
                original_weights, updated_weights
            )
            
            # Set the DP-protected weights back to the model for loss computation
            self.model.set_weights(dp_weights)
            
            # Compute training loss AFTER DP application
            train_loss_after_dp = self.model.evaluate(
                self.X_train,
                self.y_train,
                batch_size=self.batch_size,
                verbose=0
            )[0]
            
            # Use DP-protected weights for return
            final_weights = dp_weights
            
            # Log DP application
            logger.info(
                f"Client {self.client_id} applied DP: "
                f"epsilon={dp_metrics['dp_epsilon']}, "
                f"delta={dp_metrics['dp_delta']}, "
                f"l2_norm_clip={dp_metrics['dp_l2_norm_clip']}, "
                f"train_loss_after_dp={train_loss_after_dp:.4f}"
            )
        else:
            # No DP - use updated weights directly
            final_weights = updated_weights
            dp_metrics = {"dp_enabled": False}
        
        # Prepare aggregated metrics (NO patient-level data)
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history["accuracy"][-1]),
        }
        
        # Add DP metrics
        metrics.update(dp_metrics)
        
        # Add loss after DP if DP was applied
        if self.dp_mechanism is not None:
            metrics["train_loss_after_dp"] = float(train_loss_after_dp)
        
        # Add validation metrics if available
        if self.X_val is not None:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
            metrics["val_accuracy"] = float(history.history["val_accuracy"][-1])
        
        # Log training completion (aggregated metrics only)
        logger.info(
            f"Client {self.client_id} completed training: "
            f"train_loss={metrics['train_loss']:.4f}, "
            f"train_acc={metrics['train_accuracy']:.4f}"
        )
        
        # Return: DP-protected weights (if enabled), number of samples, metrics
        return final_weights, len(self.X_train), metrics
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model weights from server to evaluate
            config: Evaluation configuration from server
        
        Returns:
            Tuple containing:
                - Loss value (float)
                - Number of evaluation samples (int)
                - Evaluation metrics (Dict)
        
        Privacy Note:
            Returns only aggregated metrics - NO patient-level data.
        """
        # Set model weights
        self.set_parameters(parameters)
        
        # Use validation data if available, otherwise use training data
        X_eval = self.X_val if self.X_val is not None else self.X_train
        y_eval = self.y_val if self.y_val is not None else self.y_train
        
        # Evaluate model (verbose=0 for privacy)
        results = self.model.evaluate(
            X_eval, 
            y_eval, 
            batch_size=self.batch_size,
            verbose=0
        )
        
        # Extract loss and metrics
        loss = float(results[0])
        accuracy = float(results[1])
        
        # Prepare metrics dictionary
        metrics = {
            "accuracy": accuracy,
        }
        
        # Log evaluation (aggregated metrics only)
        logger.info(
            f"Client {self.client_id} evaluation: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}"
        )
        
        # Return: loss, number of samples, metrics
        return loss, len(X_eval), metrics


def create_flower_client(
    client_data: np.ndarray,
    preprocessor: HeartFailurePreprocessor,
    val_split: float = 0.2,
    epochs_per_round: int = 5,
    batch_size: int = 32,
    client_id: str = "unknown",
    input_shape: Tuple[int, int] = (1, 12),
    random_seed: int = 42,
    dp_config: Optional[DPConfig] = None
) -> FlowerClient:
    """
    Factory function to create a configured Flower client.
    
    This function:
    1. Preprocesses local hospital data
    2. Creates train/val split
    3. Reshapes data for LSTM input
    4. Initializes LSTM model
    5. Returns configured FlowerClient with optional DP
    
    Args:
        client_data: Raw client dataset (pandas DataFrame or numpy array)
        preprocessor: Fitted preprocessing pipeline
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        epochs_per_round: Number of local training epochs per round
        batch_size: Batch size for training
        client_id: Client identifier for logging
        input_shape: Input shape for LSTM model (sequence_length, n_features)
        random_seed: Random seed for reproducibility
        dp_config: Optional differential privacy configuration
    
    Returns:
        FlowerClient: Configured Flower client ready for federated training
    
    Privacy Note:
        All data processing happens locally. No patient data leaves the client.
        If dp_config is provided, differential privacy is applied to model updates.
    
    Example:
        >>> from utils.client_partitioning import partition_for_federated_clients
        >>> from utils.preprocessing import create_preprocessing_pipeline
        >>> from federated import create_dp_config
        >>> import pandas as pd
        >>> 
        >>> # Load and partition data
        >>> client_datasets = partition_for_federated_clients('data/heart_failure.csv', n_clients=5)
        >>> 
        >>> # Create and fit preprocessor (centrally or on aggregated data)
        >>> data = pd.read_csv('data/heart_failure.csv')
        >>> preprocessor = create_preprocessing_pipeline()
        >>> preprocessor.fit(data)
        >>> 
        >>> # Create DP config
        >>> dp_config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
        >>> 
        >>> # Create client for first hospital with DP
        >>> client = create_flower_client(
        ...     client_data=client_datasets[0],
        ...     preprocessor=preprocessor,
        ...     client_id="hospital_0",
        ...     dp_config=dp_config
        ... )
    """
    # Convert to DataFrame if needed
    if isinstance(client_data, np.ndarray):
        # Assume it's already in the correct format
        df = pd.DataFrame(client_data)
    elif isinstance(client_data, pd.DataFrame):
        df = client_data
    else:
        raise ValueError("client_data must be pandas DataFrame or numpy array")
    
    # Apply preprocessing pipeline (local only)
    X, y = preprocessor.transform(df, return_target=True)
    
    # Create train/val split
    np.random.seed(random_seed)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    if val_split > 0:
        split_idx = int(n_samples * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
    else:
        X_train = X
        y_train = y
        X_val = None
        y_val = None
    
    # Reshape for LSTM input: (batch_size, sequence_length, n_features)
    X_train = X_train.reshape(-1, input_shape[0], input_shape[1])
    if X_val is not None:
        X_val = X_val.reshape(-1, input_shape[0], input_shape[1])
    
    # Create LSTM model (PRIMARY model)
    model = get_primary_model(input_shape=input_shape)
    
    # Log client creation (NO patient data)
    logger.info(
        f"Created FlowerClient '{client_id}': "
        f"train_samples={len(X_train)}, "
        f"val_samples={len(X_val) if X_val is not None else 0}, "
        f"dp_enabled={dp_config is not None and dp_config.enabled if dp_config else False}"
    )
    
    # Create and return FlowerClient
    return FlowerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs_per_round=epochs_per_round,
        batch_size=batch_size,
        client_id=client_id,
        dp_config=dp_config
    )
