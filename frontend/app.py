"""
Flask Web Application for Heart Failure Prediction

This module implements the web interface for the inference pipeline.

Features:
- File upload interface
- CSV processing
- Prediction display
- Confidence scores
- Research disclaimer

Constraints:
- NO data storage
- NO training
- In-memory processing only

Author: Federated Learning Medical AI Project
"""

import io
import logging
import os
import sys
from typing import Dict, List

import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import InferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global inference pipeline
inference_pipeline = None


def create_app(config: Dict = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Configure app
    if config:
        app.config.update(config)
    
    # Set secret key for session
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Initialize inference pipeline
    @app.before_request
    def init_pipeline():
        """Initialize the inference pipeline on first request."""
        global inference_pipeline
        
        if inference_pipeline is None:
            logger.info("Initializing inference pipeline...")
            try:
                inference_pipeline = InferencePipeline()
                
                # Get paths relative to project root
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                history_path = os.path.join(project_root, 'logs', 'training_history.json')
                data_path = os.path.join(project_root, 'data', 'heart_failure.csv')
                weights_path = os.path.join(project_root, 'logs', 'model_weights.h5')
                
                # Try to load with saved weights
                if os.path.exists(weights_path):
                    logger.info(f"Loading model from saved weights: {weights_path}")
                    inference_pipeline.load_model_from_history(history_path, data_path, weights_path)
                else:
                    logger.warning("No saved weights found. Using fresh (untrained) model.")
                    logger.warning("For trained model, run: python save_model_weights.py")
                    
                    # Load just the preprocessor and create an untrained model
                    from models import get_primary_model
                    from utils.preprocessing import create_preprocessing_pipeline
                    import pandas as pd
                    
                    data = pd.read_csv(data_path)
                    inference_pipeline.preprocessor = create_preprocessing_pipeline()
                    inference_pipeline.preprocessor.fit(data)
                    inference_pipeline.model = get_primary_model(input_shape=(1, 12))
                    inference_pipeline.is_loaded = True
                    
                logger.info("Inference pipeline initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize inference pipeline: {e}")
                raise
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Handle prediction requests.
        
        Expects a CSV file upload with patient data.
        Returns predictions with confidence scores.
        """
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({
                    'error': 'No file uploaded'
                }), 400
            
            file = request.files['file']
            
            # Check if file has a name
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected'
                }), 400
            
            # Check file extension
            if not file.filename.endswith('.csv'):
                return jsonify({
                    'error': 'File must be a CSV file'
                }), 400
            
            # Read CSV file
            try:
                # Read file content
                file_content = file.read()
                file_stream = io.StringIO(file_content.decode('utf-8'))
                patient_data = pd.read_csv(file_stream)
            except Exception as e:
                return jsonify({
                    'error': f'Error reading CSV file: {str(e)}'
                }), 400
            
            # Validate data
            if len(patient_data) == 0:
                return jsonify({
                    'error': 'CSV file is empty'
                }), 400
            
            # Remove DEATH_EVENT column if present
            if 'DEATH_EVENT' in patient_data.columns:
                patient_data = patient_data.drop(columns=['DEATH_EVENT'])
            
            # Make predictions
            results = []
            for idx in range(len(patient_data)):
                patient_row = patient_data.iloc[idx:idx+1]
                
                try:
                    prediction, confidence = inference_pipeline.predict(patient_row)
                    
                    # Create result dictionary
                    result = {
                        'patient_id': idx + 1,
                        'prediction': prediction,
                        'confidence': round(confidence * 100, 2),  # Convert to percentage
                        'confidence_raw': round(confidence, 4)
                    }
                    
                    # Add patient features (optional, for display)
                    for col in patient_row.columns:
                        value = patient_row[col].values[0]
                        # Convert numpy types to Python types for JSON serialization
                        if hasattr(value, 'item'):
                            result[col] = value.item()
                        else:
                            result[col] = value
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error predicting for patient {idx + 1}: {e}")
                    results.append({
                        'patient_id': idx + 1,
                        'error': str(e)
                    })
            
            return jsonify({
                'success': True,
                'num_patients': len(results),
                'predictions': results
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return jsonify({
                'error': f'Prediction failed: {str(e)}'
            }), 500
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'pipeline_loaded': inference_pipeline is not None and inference_pipeline.is_loaded
        })
    
    return app


if __name__ == '__main__':
    """Run the Flask application."""
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
