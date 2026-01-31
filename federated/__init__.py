"""
Federated Learning Module for Hospital Clients

This module provides Flower federated learning client implementations
for privacy-preserving medical AI training across hospitals.

Author: Federated Learning Medical AI Project
"""

from .client import FlowerClient, create_flower_client

__all__ = [
    'FlowerClient',
    'create_flower_client',
]
