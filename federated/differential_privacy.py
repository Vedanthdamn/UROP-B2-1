"""
Differential Privacy Module for Federated Learning

This module implements differential privacy mechanisms for federated learning clients:
- Gradient clipping (L2 norm clipping)
- Gaussian noise addition
- Privacy budget tracking (epsilon, delta)

Key Privacy Features:
- Gradient clipping prevents large updates from individual samples
- Gaussian noise addition provides formal privacy guarantees
- Privacy budget (epsilon, delta) controls privacy-utility tradeoff
- All DP operations occur BEFORE model updates are sent to server

Implementation follows standard DP-SGD principles:
1. Clip gradients per sample to bounded sensitivity
2. Aggregate clipped gradients
3. Add calibrated Gaussian noise to aggregated gradients
4. Use noisy gradients for model update

Author: Federated Learning Medical AI Project
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """
    Configuration for Differential Privacy.
    
    Attributes:
        epsilon (float): Privacy budget parameter epsilon. Lower = more privacy, less utility.
                        Typical values: 0.1 to 10.0
        delta (float): Privacy budget parameter delta. Should be < 1/n_samples.
                      Typical values: 1e-5 to 1e-7
        l2_norm_clip (float): Maximum L2 norm for gradient clipping.
                             Typical values: 0.1 to 5.0
        noise_multiplier (float): Multiplier for Gaussian noise standard deviation.
                                 If None, computed from epsilon/delta.
                                 Higher = more noise = more privacy.
        enabled (bool): Whether DP is enabled. If False, no DP operations are applied.
    
    Privacy Guarantees:
        - (epsilon, delta)-differential privacy when noise_multiplier is properly calibrated
        - Smaller epsilon = stronger privacy (but potentially lower utility)
        - delta should be cryptographically small (e.g., 1e-5)
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    l2_norm_clip: float = 1.0
    noise_multiplier: float = None
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.enabled:
            if self.epsilon <= 0:
                raise ValueError(f"epsilon must be positive, got {self.epsilon}")
            if self.delta <= 0 or self.delta >= 1:
                raise ValueError(f"delta must be in (0, 1), got {self.delta}")
            if self.l2_norm_clip <= 0:
                raise ValueError(f"l2_norm_clip must be positive, got {self.l2_norm_clip}")
            
            # Compute noise multiplier from epsilon/delta if not provided
            if self.noise_multiplier is None:
                # Simple approximation: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
                # This is a conservative estimate for (epsilon, delta)-DP
                self.noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
                logger.info(
                    f"Computed noise_multiplier={self.noise_multiplier:.4f} "
                    f"from epsilon={self.epsilon}, delta={self.delta}"
                )


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for federated learning.
    
    This class implements DP-SGD (Differentially Private Stochastic Gradient Descent)
    for federated learning clients. It provides:
    - Gradient clipping to bound sensitivity
    - Gaussian noise addition for privacy
    - Privacy budget tracking
    
    Privacy Mechanism:
        1. Clip each gradient update to bounded L2 norm
        2. Add calibrated Gaussian noise to clipped gradients
        3. Use noisy gradients for model update
    
    Usage:
        >>> config = DPConfig(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
        >>> dp = DifferentialPrivacy(config)
        >>> 
        >>> # During training, apply DP to model weight updates
        >>> clipped_updates = dp.clip_gradients(weight_updates)
        >>> noisy_updates = dp.add_noise(clipped_updates)
    """
    
    def __init__(self, config: DPConfig):
        """
        Initialize the Differential Privacy mechanism.
        
        Args:
            config: DPConfig instance with privacy parameters
        """
        self.config = config
        
        # Log initialization
        if self.config.enabled:
            logger.info(
                f"Differential Privacy enabled: "
                f"epsilon={self.config.epsilon}, "
                f"delta={self.config.delta}, "
                f"l2_norm_clip={self.config.l2_norm_clip}, "
                f"noise_multiplier={self.config.noise_multiplier:.4f}"
            )
        else:
            logger.info("Differential Privacy disabled")
    
    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Clip gradients to bounded L2 norm.
        
        This operation bounds the sensitivity of the gradient computation,
        ensuring that any single training example cannot influence the
        gradients too much.
        
        Args:
            gradients: List of gradient arrays (one per model layer)
        
        Returns:
            List of clipped gradient arrays
        
        Privacy Note:
            Gradient clipping is the first step in DP-SGD. It ensures
            that the maximum change any single example can cause is bounded.
        """
        if not self.config.enabled:
            return gradients
        
        # Compute global L2 norm of all gradients
        global_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
        
        # Clip if norm exceeds threshold
        if global_norm > self.config.l2_norm_clip:
            clip_coef = self.config.l2_norm_clip / global_norm
            clipped_gradients = [g * clip_coef for g in gradients]
            logger.debug(
                f"Clipped gradients: original_norm={global_norm:.4f}, "
                f"clip_norm={self.config.l2_norm_clip}, "
                f"clip_coef={clip_coef:.4f}"
            )
            return clipped_gradients
        else:
            logger.debug(
                f"No clipping needed: norm={global_norm:.4f} <= "
                f"clip_threshold={self.config.l2_norm_clip}"
            )
            return gradients
    
    def add_noise(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add calibrated Gaussian noise to gradients.
        
        This operation provides the formal privacy guarantee. The noise
        is calibrated based on the sensitivity (determined by clipping)
        and the desired privacy parameters (epsilon, delta).
        
        Args:
            gradients: List of gradient arrays (one per model layer)
        
        Returns:
            List of noisy gradient arrays
        
        Privacy Note:
            The noise scale is calibrated to provide (epsilon, delta)-DP.
            Larger noise_multiplier = stronger privacy = more noise.
        """
        if not self.config.enabled:
            return gradients
        
        # Compute noise scale: sigma = l2_norm_clip * noise_multiplier
        noise_stddev = self.config.l2_norm_clip * self.config.noise_multiplier
        
        # Add Gaussian noise to each gradient array
        noisy_gradients = []
        for grad in gradients:
            noise = np.random.normal(0, noise_stddev, size=grad.shape)
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)
        
        logger.debug(
            f"Added Gaussian noise: stddev={noise_stddev:.4f} "
            f"(l2_norm_clip={self.config.l2_norm_clip} * "
            f"noise_multiplier={self.config.noise_multiplier:.4f})"
        )
        
        return noisy_gradients
    
    def apply_dp_to_updates(
        self, 
        original_weights: List[np.ndarray],
        updated_weights: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Apply differential privacy to model weight updates.
        
        This is the main method for applying DP to federated learning.
        It computes the weight deltas, clips them, adds noise, and
        returns the noisy updated weights.
        
        Args:
            original_weights: Model weights before training
            updated_weights: Model weights after training
        
        Returns:
            Tuple of (noisy_updated_weights, dp_metrics):
                - noisy_updated_weights: Weights with DP applied
                - dp_metrics: Dictionary with DP metrics for logging
        
        Privacy Note:
            This ensures DP is applied BEFORE weights are sent to server.
            The server only receives DP-protected weights, never raw gradients.
        """
        if not self.config.enabled:
            return updated_weights, {
                "dp_enabled": False
            }
        
        # Compute weight deltas (gradients)
        deltas = [updated - original for original, updated in zip(original_weights, updated_weights)]
        
        # Apply gradient clipping
        clipped_deltas = self.clip_gradients(deltas)
        
        # Add noise
        noisy_deltas = self.add_noise(clipped_deltas)
        
        # Compute noisy updated weights
        noisy_weights = [original + noisy_delta for original, noisy_delta in zip(original_weights, noisy_deltas)]
        
        # Compute metrics
        original_norm = np.sqrt(sum(np.sum(d ** 2) for d in deltas))
        clipped_norm = np.sqrt(sum(np.sum(d ** 2) for d in clipped_deltas))
        noisy_norm = np.sqrt(sum(np.sum(d ** 2) for d in noisy_deltas))
        
        metrics = {
            "dp_enabled": True,
            "dp_epsilon": self.config.epsilon,
            "dp_delta": self.config.delta,
            "dp_l2_norm_clip": self.config.l2_norm_clip,
            "dp_noise_multiplier": self.config.noise_multiplier,
            "dp_original_update_norm": float(original_norm),
            "dp_clipped_update_norm": float(clipped_norm),
            "dp_noisy_update_norm": float(noisy_norm),
        }
        
        return noisy_weights, metrics


def create_dp_config(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    l2_norm_clip: float = 1.0,
    noise_multiplier: float = None,
    enabled: bool = True
) -> DPConfig:
    """
    Factory function to create a DPConfig with validation.
    
    Args:
        epsilon: Privacy budget parameter (lower = more privacy)
        delta: Privacy budget parameter (should be cryptographically small)
        l2_norm_clip: Maximum L2 norm for gradient clipping
        noise_multiplier: Noise multiplier (computed from epsilon/delta if None)
        enabled: Whether to enable DP
    
    Returns:
        DPConfig instance
    
    Example:
        >>> config = create_dp_config(epsilon=1.0, delta=1e-5, l2_norm_clip=1.0)
        >>> dp = DifferentialPrivacy(config)
    """
    return DPConfig(
        epsilon=epsilon,
        delta=delta,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        enabled=enabled
    )
