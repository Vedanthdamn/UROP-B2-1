#!/usr/bin/env python3
"""
FastAPI Server for PDF-Based Disease Prediction

RESEARCH DEMONSTRATION - NOT A MEDICAL DEVICE

This module provides a REST API for offline inference using PDF blood reports.

Endpoints:
- POST /predict: Upload PDF and receive prediction
- GET /health: Health check

IMPORTANT:
- This is SEPARATE from federated learning training
- This is OFFLINE inference using a pre-trained model
- NO integration with Flower or FL rounds
- OCR/NLP are preprocessing layers, NOT part of FL

Author: UROP-B2 Team
"""

import logging
import os
import sys
import tempfile
from typing import Dict

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.inference.pdf_inference import PDFInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="PDF-based inference for heart disease prediction (Research Demo)",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
# In production, specify exact origins instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],  # Restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline
inference_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference pipeline on server startup."""
    global inference_pipeline
    
    logger.info("Initializing PDF inference pipeline...")
    
    try:
        # Initialize with default model path
        inference_pipeline = PDFInference()
        logger.info("✓ PDF inference pipeline initialized successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"✗ Model file not found: {e}")
        logger.error("Please ensure the trained model exists at backend/inference/global_model.h5")
        logger.error("Run training and save the model before starting the inference server.")
        raise
    except Exception as e:
        logger.error(f"✗ Failed to initialize inference pipeline: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "description": "PDF-based inference for heart disease prediction",
        "disclaimer": "Research demonstration only - NOT a medical device",
        "endpoints": {
            "POST /predict": "Upload PDF blood report for prediction",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status of the inference pipeline
    """
    is_ready = inference_pipeline is not None and inference_pipeline.model is not None
    
    return {
        "status": "healthy" if is_ready else "not ready",
        "inference_pipeline_loaded": is_ready,
        "model_path": inference_pipeline.model_path if inference_pipeline else None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict heart disease from uploaded PDF blood report.
    
    Args:
        file: PDF file upload
        
    Returns:
        JSON response with prediction and probability
        
    Raises:
        HTTPException: If prediction fails
    """
    # Validate inference pipeline is loaded
    if inference_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline not initialized. Server may be starting up."
        )
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF. Please upload a PDF blood report."
        )
    
    # Create temporary file to save uploaded PDF
    temp_pdf_path = None
    
    try:
        logger.info(f"Received PDF upload: {file.filename}")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_pdf_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"Saved temporary PDF to: {temp_pdf_path}")
        
        # Process PDF through inference pipeline
        label, probability, extracted_values = inference_pipeline.process_pdf(temp_pdf_path)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": label,  # "HIGH RISK" or "LOW RISK"
            "probability": round(probability, 4),
            "probability_percentage": round(probability * 100, 2),
            "extracted_values": {
                k: float(v) if v is not None else None 
                for k, v in extracted_values.items()
            },
            "disclaimer": "⚠️ This is a research demonstration, NOT a medical device. Always consult qualified healthcare professionals for medical advice."
        }
        
        logger.info(f"Prediction successful: {label} (probability: {probability:.4f})")
        
        return JSONResponse(content=response, status_code=200)
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}"
        )
    
    except ValueError as e:
        logger.error(f"Value error during prediction: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid PDF or data extraction failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
                logger.info(f"Cleaned up temporary file: {temp_pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


def main():
    """Run the FastAPI server."""
    logger.info("=" * 80)
    logger.info("PDF-BASED DISEASE PREDICTION API SERVER")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Starting FastAPI server...")
    logger.info("API will be available at: http://localhost:8000")
    logger.info("API documentation at: http://localhost:8000/docs")
    logger.info("")
    logger.info("⚠️  RESEARCH DEMONSTRATION ONLY - NOT A MEDICAL DEVICE")
    logger.info("")
    logger.info("=" * 80)
    
    # Run server with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
