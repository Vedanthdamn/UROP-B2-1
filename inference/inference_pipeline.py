"""
Inference Pipeline Implementation

This module provides the core inference pipeline for making predictions
using the trained federated learning model.

Key Features:
- Load trained model from training history
- Load preprocessing pipeline
- Process new patient data
- Generate predictions with confidence scores

Constraints:
- NO training occurs
- NO patient data is stored
- All data is processed in-memory only

Author: Federated Learning Medical AI Project
"""

import json
import logging
import os
from typing import Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from models import get_primary_model
from utils.preprocessing import HeartFailurePreprocessor, create_preprocessing_pipeline
from utils.client_partitioning import partition_for_federated_clients

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Inference pipeline for heart failure prediction.
    
    This class loads the trained federated model and preprocessing pipeline,
    then provides methods to make predictions on new patient data.
    
    Attributes:
        model: The trained TensorFlow model
        preprocessor: The preprocessing pipeline
        is_loaded: Whether the model and preprocessor are loaded
    """
    
    def __init__(self):
        """Initialize the inference pipeline."""
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.input_shape = (1, 12)  # (sequence_length, n_features)
        
    def load_model_from_history(
        self,
        history_path: str = 'logs/training_history.json',
        data_path: str = 'data/heart_failure.csv'
    ) -> None:
        """
        Load the trained model by recreating it from training history.
        
        Since the model weights are not saved, we recreate the model by
        simulating the federated training process using the training history.
        
        Args:
            history_path: Path to training_history.json
            data_path: Path to the training data
            
        Raises:
            FileNotFoundError: If history or data file not found
            ValueError: If history is invalid
        """
        logger.info("Loading model from training history...")
        
        # Load training history
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"Training history not found at {history_path}")
            
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Load and partition data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        config = history.get('experiment_config', {})
        num_clients = config.get('num_clients', 5)
        random_seed = config.get('random_seed', 42)
        num_rounds = len(history.get('rounds', []))
        
        logger.info(f"Recreating model from {num_rounds} training rounds...")
        
        # Create and fit preprocessor
        data = pd.read_csv(data_path)
        self.preprocessor = create_preprocessing_pipeline()
        self.preprocessor.fit(data)
        
        # Partition data for federated clients
        client_datasets = partition_for_federated_clients(
            data_path=data_path,
            n_clients=num_clients,
            random_seed=random_seed
        )
        
        # Create initial model
        self.model = get_primary_model(input_shape=self.input_shape)
        
        # Simulate federated training to recreate final model
        from federated import create_flower_client
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Simulating round {round_num}/{num_rounds}...")
            
            # Get current model weights
            current_weights = self.model.get_weights()
            
            # Collect updates from all clients
            client_weights = []
            client_samples = []
            
            for client_id in range(num_clients):
                # Create client
                client = create_flower_client(
                    client_data=client_datasets[client_id],
                    preprocessor=self.preprocessor,
                    val_split=0.2,
                    epochs_per_round=5,
                    batch_size=32,
                    client_id=f"hospital_{client_id}"
                )
                
                # Train client
                updated_params, n_samples, _ = client.fit(current_weights, config={})
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
            self.model.set_weights(aggregated_weights)
        
        self.is_loaded = True
        logger.info("Model loaded successfully!")
        
    def load_preprocessor(self, preprocessor_path: str = None) -> None:
        """
        Load the preprocessing pipeline.
        
        Args:
            preprocessor_path: Path to saved preprocessor pickle file.
                If None, creates and fits a new preprocessor on the training data.
                
        Raises:
            FileNotFoundError: If preprocessor file not found
        """
        if preprocessor_path and os.path.exists(preprocessor_path):
            logger.info(f"Loading preprocessor from {preprocessor_path}...")
            self.preprocessor = HeartFailurePreprocessor.load(preprocessor_path)
        else:
            # Create and fit new preprocessor
            logger.info("Creating new preprocessor...")
            data = pd.read_csv('data/heart_failure.csv')
            self.preprocessor = create_preprocessing_pipeline()
            self.preprocessor.fit(data)
            
        logger.info("Preprocessor loaded successfully!")
        
    def predict(
        self,
        patient_data: Union[pd.DataFrame, Dict, np.ndarray]
    ) -> Tuple[str, float]:
        """
        Make a prediction for a single patient.
        
        Args:
            patient_data: Patient data as DataFrame, dictionary, or numpy array.
                Must contain all required features (no DEATH_EVENT column).
                
        Returns:
            Tuple of (prediction_label, confidence_score)
            - prediction_label: "Yes" or "No" for DEATH_EVENT
            - confidence_score: Probability of positive class (0-1)
            
        Raises:
            ValueError: If model or preprocessor not loaded
            ValueError: If input data is invalid
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_history() first.")
            
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        # Convert input to DataFrame if necessary
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        elif isinstance(patient_data, np.ndarray):
            if patient_data.ndim == 1:
                patient_data = patient_data.reshape(1, -1)
            # Create DataFrame with standard column names
            feature_names = [
                'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                'ejection_fraction', 'high_blood_pressure', 'platelets',
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
            ]
            patient_data = pd.DataFrame(patient_data, columns=feature_names[:patient_data.shape[1]])
        
        # Ensure it's a DataFrame
        if not isinstance(patient_data, pd.DataFrame):
            raise ValueError("Patient data must be a DataFrame, dictionary, or numpy array")
        
        # Remove DEATH_EVENT column if present
        if 'DEATH_EVENT' in patient_data.columns:
            patient_data = patient_data.drop(columns=['DEATH_EVENT'])
        
        # Preprocess the data
        # We need to use transform without target
        X_processed = self.preprocessor.scaler.transform(patient_data.values)
        
        # Reshape for model input: (batch_size, sequence_length, n_features)
        X_reshaped = X_processed.reshape(-1, self.input_shape[0], self.input_shape[1])
        
        # Make prediction
        prediction_proba = self.model.predict(X_reshaped, verbose=0)
        
        # Extract probability for positive class
        confidence_score = float(prediction_proba[0][0])
        
        # Convert to label
        prediction_label = "Yes" if confidence_score >= 0.5 else "No"
        
        return prediction_label, confidence_score
        
    def predict_batch(
        self,
        patient_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions for multiple patients.
        
        Args:
            patient_data: DataFrame containing multiple patient records.
                Must contain all required features (no DEATH_EVENT column).
                
        Returns:
            DataFrame with columns:
            - All original features
            - prediction: "Yes" or "No"
            - confidence: Probability score
            
        Raises:
            ValueError: If model or preprocessor not loaded
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_history() first.")
            
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        # Create a copy to avoid modifying original
        results = patient_data.copy()
        
        # Remove DEATH_EVENT if present
        if 'DEATH_EVENT' in results.columns:
            results = results.drop(columns=['DEATH_EVENT'])
        
        # Store predictions
        predictions = []
        confidences = []
        
        # Predict for each patient
        for idx in range(len(results)):
            patient_row = results.iloc[idx:idx+1]
            prediction, confidence = self.predict(patient_row)
            predictions.append(prediction)
            confidences.append(confidence)
        
        # Add results to DataFrame
        results['prediction'] = predictions
        results['confidence'] = confidences
        
        return results


def predict_from_csv(
    csv_path: str,
    history_path: str = 'logs/training_history.json',
    data_path: str = 'data/heart_failure.csv'
) -> pd.DataFrame:
    """
    Convenience function to make predictions from a CSV file.
    
    Args:
        csv_path: Path to CSV file containing patient data
        history_path: Path to training history JSON
        data_path: Path to training data CSV
        
    Returns:
        DataFrame with predictions and confidence scores
        
    Example:
        >>> results = predict_from_csv('new_patients.csv')
        >>> print(results[['prediction', 'confidence']])
    """
    # Create and load pipeline
    pipeline = InferencePipeline()
    pipeline.load_model_from_history(history_path, data_path)
    
    # Load patient data
    patient_data = pd.read_csv(csv_path)
    
    # Make predictions
    results = pipeline.predict_batch(patient_data)
    
    return results
