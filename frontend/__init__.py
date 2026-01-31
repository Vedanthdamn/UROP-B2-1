"""
Frontend Web Interface for Heart Failure Prediction

This module provides a simple web interface for uploading patient data
and viewing predictions from the trained federated model.

Features:
- Simple web interface with file upload
- Display predicted disease (DEATH_EVENT: Yes/No)
- Display confidence score (probability)
- Research disclaimer

Constraints:
- NO training occurs
- NO patient data is stored
- Predictions are made in-memory only

Author: Federated Learning Medical AI Project
"""

from .app import create_app

__all__ = ['create_app']
