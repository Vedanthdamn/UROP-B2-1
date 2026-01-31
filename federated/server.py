"""
Flower Federated Server for Coordinating Training

This module implements the Flower federated learning server for coordinating
privacy-preserving training across multiple hospital clients.

Key Features:
- Uses Flower server APIs for orchestration
- Initializes global TensorFlow LSTM model
- Implements FedAvg (Federated Averaging) as primary aggregation strategy
- Supports FedProx as optional comparative strategy
- Tracks and logs per-round metrics (loss, accuracy, participating clients)

Server Responsibilities:
- Distribute global model parameters to clients
- Aggregate DP-protected client updates
- Track training progress across rounds
- NEVER access raw client data

Privacy Guarantees:
- Server operates only on model weights
- All client updates are treated as DP-protected
- No access to raw patient data
- Only aggregated metrics are logged

Author: Federated Learning Medical AI Project
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg, FedProx
from flwr.server.client_proxy import ClientProxy

from models import get_primary_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Flower federated learning server for hospital coordination.
    
    This server orchestrates federated training across multiple hospitals,
    aggregating DP-protected model updates while ensuring privacy.
    
    Key Features:
    - Global model initialization and distribution
    - Configurable aggregation strategies (FedAvg, FedProx)
    - Per-round metrics tracking and logging
    - Privacy-preserving: No access to raw client data
    
    Attributes:
        strategy_name (str): Name of aggregation strategy ('fedavg' or 'fedprox')
        strategy (fl.server.strategy.Strategy): Flower aggregation strategy
        num_rounds (int): Number of federated training rounds
        fraction_fit (float): Fraction of clients to sample for training
        fraction_evaluate (float): Fraction of clients to sample for evaluation
        min_fit_clients (int): Minimum clients required for training
        min_evaluate_clients (int): Minimum clients required for evaluation
        min_available_clients (int): Minimum clients required to start
        input_shape (tuple): Input shape for LSTM model
    """
    
    def __init__(
        self,
        strategy_name: str = "fedavg",
        num_rounds: int = 10,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        input_shape: Tuple[int, int] = (1, 12),
        proximal_mu: float = 0.1,
        **kwargs
    ):
        """
        Initialize the federated server.
        
        Args:
            strategy_name: Aggregation strategy ('fedavg' or 'fedprox')
            num_rounds: Number of federated training rounds
            fraction_fit: Fraction of clients to sample for training per round
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of clients to start training
            input_shape: Input shape for LSTM model (sequence_length, n_features)
            proximal_mu: Proximal term parameter for FedProx (only used if strategy_name='fedprox')
            **kwargs: Additional arguments passed to strategy
        
        Privacy Note:
            Server only handles model weights, never raw patient data.
            All client updates are treated as DP-protected.
        """
        self.strategy_name = strategy_name.lower()
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.input_shape = input_shape
        self.proximal_mu = proximal_mu
        
        # Initialize global model (PRIMARY LSTM model)
        logger.info("Initializing global TensorFlow LSTM model...")
        self.global_model = get_primary_model(input_shape=input_shape)
        logger.info(f"Global model initialized with {self.global_model.count_params():,} parameters")
        
        # Get initial model parameters
        initial_parameters = ndarrays_to_parameters(self.global_model.get_weights())
        
        # Create aggregation strategy
        self.strategy = self._create_strategy(initial_parameters, **kwargs)
        
        logger.info(
            f"Federated server initialized: "
            f"strategy={self.strategy_name}, "
            f"num_rounds={num_rounds}, "
            f"min_clients={min_available_clients}"
        )
    
    def _create_strategy(self, initial_parameters: Parameters, **kwargs) -> fl.server.strategy.Strategy:
        """
        Create the aggregation strategy.
        
        Args:
            initial_parameters: Initial global model parameters
            **kwargs: Additional strategy arguments
        
        Returns:
            Flower aggregation strategy (FedAvg or FedProx)
        """
        # Common strategy configuration
        strategy_config = {
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "initial_parameters": initial_parameters,
            "fit_metrics_aggregation_fn": self._aggregate_fit_metrics,
            "evaluate_metrics_aggregation_fn": self._aggregate_evaluate_metrics,
        }
        
        # Add any additional kwargs
        strategy_config.update(kwargs)
        
        # Create strategy based on name
        if self.strategy_name == "fedavg":
            strategy = FedAvg(**strategy_config)
            logger.info("Created FedAvg aggregation strategy (primary)")
        elif self.strategy_name == "fedprox":
            # FedProx requires proximal_mu parameter
            strategy_config["proximal_mu"] = self.proximal_mu
            strategy = FedProx(**strategy_config)
            logger.info(f"Created FedProx aggregation strategy (comparative, mu={self.proximal_mu})")
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. "
                f"Supported strategies: 'fedavg', 'fedprox'"
            )
        
        return strategy
    
    def _aggregate_fit_metrics(
        self, 
        metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """
        Aggregate training metrics from clients.
        
        This method aggregates metrics reported by clients after local training.
        Computes weighted averages based on number of samples per client.
        
        Args:
            metrics: List of tuples (num_samples, metrics_dict) from each client
        
        Returns:
            Dictionary of aggregated metrics
        
        Privacy Note:
            Only aggregated metrics are logged, no client-specific data.
        """
        if not metrics:
            return {}
        
        # Calculate total samples
        total_samples = sum(num_samples for num_samples, _ in metrics)
        
        # Initialize accumulators
        aggregated = {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "num_clients": len(metrics),
            "total_samples": total_samples,
        }
        
        # Weighted aggregation
        for num_samples, client_metrics in metrics:
            weight = num_samples / total_samples
            
            # Aggregate basic metrics
            if "train_loss" in client_metrics:
                aggregated["train_loss"] += float(client_metrics["train_loss"]) * weight
            if "train_accuracy" in client_metrics:
                aggregated["train_accuracy"] += float(client_metrics["train_accuracy"]) * weight
            
            # Track DP usage (any client using DP)
            if "dp_enabled" in client_metrics and client_metrics["dp_enabled"]:
                aggregated["dp_enabled"] = True
                if "dp_epsilon" in client_metrics:
                    # Note: In practice, privacy budgets compose across rounds.
                    # This is just logging the epsilon value from clients for reference.
                    # For proper privacy accounting in multi-round federated learning:
                    # 1. Use advanced composition theorems (basic, optimal, or Rényi DP)
                    # 2. Consider using privacy accounting libraries like:
                    #    - TensorFlow Privacy's privacy_accountant
                    #    - Google's differential-privacy library
                    # 3. Track cumulative privacy loss across all rounds
                    # WARNING: The current epsilon value does NOT reflect composed privacy
                    # budget. Total privacy budget degrades with each round of training.
                    aggregated["dp_epsilon"] = float(client_metrics["dp_epsilon"])
                if "dp_delta" in client_metrics:
                    aggregated["dp_delta"] = float(client_metrics["dp_delta"])
        
        # Log aggregated metrics (privacy-safe: only aggregates)
        logger.info(
            f"Round completed: "
            f"clients={aggregated['num_clients']}, "
            f"samples={aggregated['total_samples']}, "
            f"avg_loss={aggregated['train_loss']:.4f}, "
            f"avg_accuracy={aggregated['train_accuracy']:.4f}, "
            f"dp_enabled={aggregated.get('dp_enabled', False)}"
        )
        
        return aggregated
    
    def _aggregate_evaluate_metrics(
        self, 
        metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """
        Aggregate evaluation metrics from clients.
        
        This method aggregates metrics reported by clients after evaluation.
        Computes weighted averages based on number of samples per client.
        
        Args:
            metrics: List of tuples (num_samples, metrics_dict) from each client
        
        Returns:
            Dictionary of aggregated metrics
        
        Privacy Note:
            Only aggregated metrics are logged, no client-specific data.
        """
        if not metrics:
            return {}
        
        # Calculate total samples
        total_samples = sum(num_samples for num_samples, _ in metrics)
        
        # Initialize accumulators
        aggregated = {
            "eval_accuracy": 0.0,
            "num_clients": len(metrics),
            "total_samples": total_samples,
        }
        
        # Weighted aggregation
        for num_samples, client_metrics in metrics:
            weight = num_samples / total_samples
            
            if "accuracy" in client_metrics:
                aggregated["eval_accuracy"] += float(client_metrics["accuracy"]) * weight
        
        # Log aggregated metrics (privacy-safe: only aggregates)
        logger.info(
            f"Evaluation completed: "
            f"clients={aggregated['num_clients']}, "
            f"samples={aggregated['total_samples']}, "
            f"avg_accuracy={aggregated['eval_accuracy']:.4f}"
        )
        
        return aggregated
    
    def start(
        self,
        server_address: str = "0.0.0.0:8080",
        config: Optional[fl.server.ServerConfig] = None
    ):
        """
        Start the federated server.
        
        This method starts the Flower server and begins accepting client connections.
        The server will run for the specified number of rounds, coordinating
        training across connected clients.
        
        Args:
            server_address: Server address in format "host:port"
            config: Optional Flower ServerConfig. If None, uses default config
                   with self.num_rounds.
        
        Returns:
            History object containing training metrics per round
        
        Privacy Note:
            Server only handles model aggregation, never accesses raw client data.
        
        Example:
            >>> server = FederatedServer(strategy_name="fedavg", num_rounds=10)
            >>> history = server.start(server_address="0.0.0.0:8080")
        """
        if config is None:
            config = fl.server.ServerConfig(num_rounds=self.num_rounds)
        
        logger.info(
            f"Starting federated server at {server_address} "
            f"for {config.num_rounds} rounds..."
        )
        logger.info(
            f"Waiting for at least {self.min_available_clients} clients to connect..."
        )
        
        # Start Flower server
        history = fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=self.strategy,
        )
        
        logger.info("Federated training completed!")
        
        return history


def create_federated_server(
    strategy: str = "fedavg",
    num_rounds: int = 10,
    min_clients: int = 2,
    input_shape: Tuple[int, int] = (1, 12),
    proximal_mu: float = 0.1,
    **kwargs
) -> FederatedServer:
    """
    Factory function to create a configured federated server.
    
    Args:
        strategy: Aggregation strategy ('fedavg' or 'fedprox')
        num_rounds: Number of federated training rounds
        min_clients: Minimum number of clients required
        input_shape: Input shape for LSTM model
        proximal_mu: Proximal term for FedProx (if strategy='fedprox')
        **kwargs: Additional arguments passed to FederatedServer
    
    Returns:
        Configured FederatedServer ready to start
    
    Privacy Note:
        Server operates only on model weights, treating all client
        updates as DP-protected. No access to raw patient data.
    
    Example:
        >>> # Create server with FedAvg
        >>> server = create_federated_server(
        ...     strategy="fedavg",
        ...     num_rounds=10,
        ...     min_clients=3
        ... )
        >>> history = server.start(server_address="0.0.0.0:8080")
        
        >>> # Create server with FedProx
        >>> server = create_federated_server(
        ...     strategy="fedprox",
        ...     num_rounds=10,
        ...     min_clients=3,
        ...     proximal_mu=0.1
        ... )
        >>> history = server.start(server_address="0.0.0.0:8080")
    """
    return FederatedServer(
        strategy_name=strategy,
        num_rounds=num_rounds,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        input_shape=input_shape,
        proximal_mu=proximal_mu,
        **kwargs
    )


def start_server_simulation(
    client_fn: Callable[[str], fl.client.Client],
    num_clients: int = 5,
    strategy: str = "fedavg",
    num_rounds: int = 10,
    input_shape: Tuple[int, int] = (1, 12),
    proximal_mu: float = 0.1,
    **kwargs
) -> fl.server.history.History:
    """
    Start federated learning in simulation mode.
    
    This function runs federated learning in simulation mode, where all clients
    run locally in the same process. This is useful for:
    - Development and testing
    - Experimentation without network setup
    - Quick prototyping
    
    Args:
        client_fn: Function that creates a Flower client given a client ID
        num_clients: Total number of clients to simulate
        strategy: Aggregation strategy ('fedavg' or 'fedprox')
        num_rounds: Number of federated training rounds
        input_shape: Input shape for LSTM model
        proximal_mu: Proximal term for FedProx (if strategy='fedprox')
        **kwargs: Additional arguments passed to strategy
    
    Returns:
        History object containing training metrics per round
    
    Privacy Note:
        Even in simulation, clients only share model weights, not raw data.
    
    Example:
        >>> from federated import create_flower_client
        >>> from utils.client_partitioning import partition_for_federated_clients
        >>> from utils.preprocessing import create_preprocessing_pipeline
        >>> import pandas as pd
        >>> 
        >>> # Prepare data and preprocessor
        >>> client_datasets = partition_for_federated_clients('data/heart_failure.csv', n_clients=5)
        >>> data = pd.read_csv('data/heart_failure.csv')
        >>> preprocessor = create_preprocessing_pipeline()
        >>> preprocessor.fit(data)
        >>> 
        >>> # Define client factory
        >>> def client_fn(cid: str):
        ...     client_id = int(cid)
        ...     return create_flower_client(
        ...         client_data=client_datasets[client_id],
        ...         preprocessor=preprocessor,
        ...         client_id=f"hospital_{client_id}"
        ...     )
        >>> 
        >>> # Run simulation
        >>> history = start_server_simulation(
        ...     client_fn=client_fn,
        ...     num_clients=5,
        ...     strategy="fedavg",
        ...     num_rounds=10
        ... )
    """
    logger.info(
        f"Starting federated simulation: "
        f"clients={num_clients}, "
        f"strategy={strategy}, "
        f"rounds={num_rounds}"
    )
    
    # Initialize global model
    global_model = get_primary_model(input_shape=input_shape)
    initial_parameters = ndarrays_to_parameters(global_model.get_weights())
    
    # Create aggregation strategy
    strategy_config = {
        "fraction_fit": 1.0,  # Use all clients in simulation
        "fraction_evaluate": 1.0,
        "min_fit_clients": min(2, num_clients),
        "min_evaluate_clients": min(2, num_clients),
        "min_available_clients": min(2, num_clients),
        "initial_parameters": initial_parameters,
    }
    
    # Add custom metric aggregation
    server = FederatedServer(
        strategy_name=strategy,
        num_rounds=num_rounds,
        input_shape=input_shape,
        proximal_mu=proximal_mu,
    )
    
    strategy_config["fit_metrics_aggregation_fn"] = server._aggregate_fit_metrics
    strategy_config["evaluate_metrics_aggregation_fn"] = server._aggregate_evaluate_metrics
    strategy_config.update(kwargs)
    
    if strategy.lower() == "fedavg":
        fl_strategy = FedAvg(**strategy_config)
    elif strategy.lower() == "fedprox":
        strategy_config["proximal_mu"] = proximal_mu
        fl_strategy = FedProx(**strategy_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl_strategy,
    )
    
    logger.info("Federated simulation completed!")
    
    return history
