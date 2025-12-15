"""Uncertainty Estimation for epistemic and aleatoric uncertainty."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class UncertaintyEstimator:
    """Estimates epistemic and aleatoric uncertainty in predictions.
    
    Epistemic uncertainty: Model uncertainty due to limited data/knowledge
    Aleatoric uncertainty: Data uncertainty inherent in the environment
    """

    def __init__(
        self,
        method: str = "ensemble",
        num_samples: int = 10,
        confidence_threshold: float = 0.8
    ):
        """Initialize uncertainty estimator.
        
        Args:
            method: Uncertainty estimation method ('ensemble', 'dropout', 'bayesian')
            num_samples: Number of samples for Monte Carlo estimation
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.method = method
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold

    def estimate(
        self,
        predictions: torch.Tensor,
        model: Optional[nn.Module] = None,
        input_data: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty in predictions.
        
        Args:
            predictions: Model predictions [num_samples, batch_size, ...]
            model: Optional model for MC dropout
            input_data: Optional input for MC dropout
            
        Returns:
            Dictionary with uncertainty metrics
        """
        if self.method == "ensemble":
            return self._ensemble_uncertainty(predictions)
        elif self.method == "dropout" and model is not None and input_data is not None:
            return self._dropout_uncertainty(model, input_data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _ensemble_uncertainty(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute uncertainty from ensemble predictions.
        
        Args:
            predictions: Predictions from ensemble [num_models, batch_size, ...]
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Variance (epistemic uncertainty)
        variance = predictions.var(dim=0)
        std = torch.sqrt(variance)
        
        # Confidence (inverse of normalized uncertainty)
        normalized_std = std / (predictions.abs().mean() + 1e-8)
        confidence = 1.0 / (1.0 + normalized_std)
        
        # Disagreement (max - min)
        disagreement = predictions.max(dim=0)[0] - predictions.min(dim=0)[0]
        
        return {
            "mean": mean_pred,
            "std": std,
            "variance": variance,
            "confidence": confidence,
            "disagreement": disagreement,
            "epistemic_uncertainty": variance,
        }

    def _dropout_uncertainty(
        self,
        model: nn.Module,
        input_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty using MC Dropout.
        
        Args:
            model: Model with dropout layers
            input_data: Input data for prediction
            
        Returns:
            Dictionary with uncertainty metrics
        """
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = model(input_data)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        model.eval()  # Disable dropout
        
        return self._ensemble_uncertainty(predictions)

    def compute_information_gain(
        self,
        prior_uncertainty: torch.Tensor,
        posterior_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Compute information gain from taking an action.
        
        Args:
            prior_uncertainty: Uncertainty before action
            posterior_uncertainty: Uncertainty after action
            
        Returns:
            Information gain tensor
        """
        # Information gain as reduction in entropy
        return torch.clamp(prior_uncertainty - posterior_uncertainty, min=0.0)

    def is_high_confidence(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Check if predictions are high confidence.
        
        Args:
            uncertainty: Uncertainty tensor
            
        Returns:
            Boolean tensor indicating high confidence
        """
        confidence = 1.0 / (1.0 + uncertainty)
        return confidence >= self.confidence_threshold

    def calibrate_uncertainty(
        self,
        predicted_std: torch.Tensor,
        actual_errors: torch.Tensor
    ) -> Tuple[float, float]:
        """Calibrate uncertainty estimates using observed errors.
        
        Args:
            predicted_std: Predicted standard deviations
            actual_errors: Actual prediction errors
            
        Returns:
            Tuple of (calibration_scale, calibration_bias)
        """
        # Compute calibration using least squares
        # actual_errors ≈ scale * predicted_std + bias
        
        predicted_std_np = predicted_std.detach().cpu().numpy().flatten()
        actual_errors_np = actual_errors.detach().cpu().numpy().flatten()
        
        # Least squares fit
        A = np.vstack([predicted_std_np, np.ones(len(predicted_std_np))]).T
        scale, bias = np.linalg.lstsq(A, actual_errors_np, rcond=None)[0]
        
        return float(scale), float(bias)


class BayesianUncertaintyEstimator(UncertaintyEstimator):
    """Bayesian uncertainty estimation using variational inference."""

    def __init__(
        self,
        prior_scale: float = 1.0,
        num_samples: int = 10,
        confidence_threshold: float = 0.8
    ):
        """Initialize Bayesian uncertainty estimator.
        
        Args:
            prior_scale: Scale of prior distribution
            num_samples: Number of samples for Monte Carlo estimation
            confidence_threshold: Threshold for high-confidence predictions
        """
        super().__init__("bayesian", num_samples, confidence_threshold)
        self.prior_scale = prior_scale

    def kl_divergence(
        self,
        posterior_mean: torch.Tensor,
        posterior_std: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and prior.
        
        Args:
            posterior_mean: Mean of posterior distribution
            posterior_std: Std of posterior distribution
            
        Returns:
            KL divergence
        """
        # KL(q||p) for Gaussian q and standard Gaussian prior p
        kl = 0.5 * (
            posterior_std**2 / self.prior_scale**2 +
            posterior_mean**2 / self.prior_scale**2 -
            1.0 -
            torch.log(posterior_std**2 / self.prior_scale**2)
        )
        return kl.sum()


class ActiveLearningSelector:
    """Select informative samples for active learning based on uncertainty."""

    def __init__(self, strategy: str = "max_entropy"):
        """Initialize active learning selector.
        
        Args:
            strategy: Selection strategy ('max_entropy', 'bald', 'random')
        """
        self.strategy = strategy

    def select_samples(
        self,
        uncertainty: torch.Tensor,
        num_samples: int,
        already_selected: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Select most informative samples.
        
        Args:
            uncertainty: Uncertainty scores [batch_size]
            num_samples: Number of samples to select
            already_selected: Mask of already selected samples
            
        Returns:
            Indices of selected samples
        """
        if self.strategy == "max_entropy":
            scores = uncertainty
        elif self.strategy == "random":
            scores = torch.rand_like(uncertainty)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Mask already selected samples
        if already_selected is not None:
            scores = scores * (1 - already_selected)
        
        # Select top-k
        _, indices = torch.topk(scores, k=min(num_samples, len(scores)))
        
        return indices
