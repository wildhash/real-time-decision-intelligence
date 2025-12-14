"""Tests for uncertainty estimation module."""

import torch
import pytest
import numpy as np
from rtdi.uncertainty import (
    UncertaintyEstimator,
    BayesianUncertaintyEstimator,
    ActiveLearningSelector
)


class TestUncertaintyEstimator:
    """Test suite for UncertaintyEstimator."""

    def setup_method(self):
        """Setup for each test."""
        self.batch_size = 16
        self.output_dim = 10
        self.num_ensemble = 5
        
    def test_initialization_ensemble(self):
        """Test ensemble estimator initialization."""
        estimator = UncertaintyEstimator(method="ensemble", num_samples=10)
        
        assert estimator.method == "ensemble"
        assert estimator.num_samples == 10
        assert estimator.confidence_threshold == 0.8
        
    def test_initialization_dropout(self):
        """Test dropout estimator initialization."""
        estimator = UncertaintyEstimator(method="dropout", num_samples=20)
        
        assert estimator.method == "dropout"
        assert estimator.num_samples == 20
        
    def test_ensemble_uncertainty_estimation(self):
        """Test uncertainty estimation from ensemble predictions."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Create ensemble predictions
        predictions = torch.randn(self.num_ensemble, self.batch_size, self.output_dim)
        
        result = estimator.estimate(predictions)
        
        # Check output keys
        assert "mean" in result
        assert "std" in result
        assert "variance" in result
        assert "confidence" in result
        assert "disagreement" in result
        assert "epistemic_uncertainty" in result
        
        # Check shapes
        assert result["mean"].shape == (self.batch_size, self.output_dim)
        assert result["std"].shape == (self.batch_size, self.output_dim)
        assert result["variance"].shape == (self.batch_size, self.output_dim)
        
        # Check values
        assert (result["std"] >= 0).all()
        assert (result["variance"] >= 0).all()
        assert (result["confidence"] > 0).all()
        assert (result["confidence"] <= 1.0).all()
        
    def test_ensemble_mean_calculation(self):
        """Test that ensemble mean is calculated correctly."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Create known predictions
        predictions = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[3.0, 4.0], [5.0, 6.0]]
        ])  # [3, 2, 2]
        
        result = estimator.estimate(predictions)
        
        # Expected mean: [2.0, 3.0], [4.0, 5.0]
        expected_mean = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        
        assert torch.allclose(result["mean"], expected_mean, atol=1e-5)
        
    def test_high_confidence_detection(self):
        """Test high confidence detection."""
        estimator = UncertaintyEstimator(method="ensemble", confidence_threshold=0.8)
        
        # Low uncertainty (high confidence)
        low_uncertainty = torch.tensor([[0.1, 0.2], [0.15, 0.1]])
        is_confident = estimator.is_high_confidence(low_uncertainty)
        
        assert is_confident.any()
        
        # High uncertainty (low confidence)
        high_uncertainty = torch.tensor([[10.0, 15.0], [20.0, 25.0]])
        is_not_confident = estimator.is_high_confidence(high_uncertainty)
        
        assert not is_not_confident.all()
        
    def test_information_gain_computation(self):
        """Test information gain computation."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        prior_uncertainty = torch.tensor([1.0, 2.0, 3.0])
        posterior_uncertainty = torch.tensor([0.5, 1.0, 2.5])
        
        info_gain = estimator.compute_information_gain(prior_uncertainty, posterior_uncertainty)
        
        # Information gain should be non-negative
        assert (info_gain >= 0).all()
        
        # Expected gains: [0.5, 1.0, 0.5]
        expected = torch.tensor([0.5, 1.0, 0.5])
        assert torch.allclose(info_gain, expected, atol=1e-5)
        
    def test_information_gain_no_reduction(self):
        """Test information gain when uncertainty doesn't reduce."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        prior_uncertainty = torch.tensor([1.0, 2.0])
        posterior_uncertainty = torch.tensor([1.5, 2.5])  # Increased
        
        info_gain = estimator.compute_information_gain(prior_uncertainty, posterior_uncertainty)
        
        # Should clamp negative values to zero
        assert (info_gain >= 0).all()
        assert torch.allclose(info_gain, torch.zeros_like(info_gain))
        
    def test_uncertainty_calibration(self):
        """Test uncertainty calibration."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Predicted uncertainties
        predicted_std = torch.tensor([0.5, 1.0, 1.5, 2.0])
        
        # Actual errors
        actual_errors = torch.tensor([0.6, 0.9, 1.6, 1.8])
        
        calibration_error, scaling_factor = estimator.calibrate_uncertainty(
            predicted_std, actual_errors
        )
        
        assert isinstance(calibration_error, float)
        assert isinstance(scaling_factor, float)
        assert calibration_error >= 0
        assert scaling_factor > 0
        
    def test_disagreement_metric(self):
        """Test disagreement calculation in ensemble."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Create predictions with known disagreement
        predictions = torch.tensor([
            [[0.0, 5.0]],
            [[1.0, 6.0]],
            [[2.0, 7.0]]
        ])  # [3, 1, 2]
        
        result = estimator.estimate(predictions)
        
        # Disagreement should be max - min = [2.0, 2.0]
        expected_disagreement = torch.tensor([[2.0, 2.0]])
        
        assert torch.allclose(result["disagreement"], expected_disagreement, atol=1e-5)
        
    def test_dropout_uncertainty_with_model(self):
        """Test MC Dropout uncertainty estimation."""
        import torch.nn as nn
        
        estimator = UncertaintyEstimator(method="dropout", num_samples=10)
        
        # Create simple model with dropout
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.Dropout(0.5),
            nn.Linear(10, 3)
        )
        
        input_data = torch.randn(8, 5)
        
        result = estimator.estimate(predictions=None, model=model, input_data=input_data)
        
        # Check output structure
        assert "mean" in result
        assert "std" in result
        assert "variance" in result
        assert result["mean"].shape == (8, 3)
        
    def test_unsupported_method_error(self):
        """Test error handling for unsupported methods."""
        estimator = UncertaintyEstimator(method="invalid_method")
        
        predictions = torch.randn(5, 10, 3)
        
        with pytest.raises(ValueError):
            estimator.estimate(predictions)
            
    def test_batch_uncertainty_estimation(self):
        """Test uncertainty estimation with different batch sizes."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Small batch
        predictions_small = torch.randn(5, 2, 10)
        result_small = estimator.estimate(predictions_small)
        assert result_small["mean"].shape == (2, 10)
        
        # Large batch
        predictions_large = torch.randn(5, 100, 10)
        result_large = estimator.estimate(predictions_large)
        assert result_large["mean"].shape == (100, 10)
        
    def test_single_prediction_uncertainty(self):
        """Test uncertainty with single prediction (no ensemble)."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Single model prediction
        predictions = torch.randn(1, 10, 5)
        result = estimator.estimate(predictions)
        
        # Variance should be zero or very small
        assert (result["variance"] < 1e-6).all()


class TestBayesianUncertaintyEstimator:
    """Test suite for BayesianUncertaintyEstimator."""

    def test_initialization(self):
        """Test Bayesian estimator initialization."""
        estimator = BayesianUncertaintyEstimator(num_samples=20)
        
        assert estimator.num_samples == 20
        assert estimator.method == "bayesian"
        
    def test_estimate_with_bayesian_model(self):
        """Test estimation with Bayesian model."""
        import torch.nn as nn
        
        estimator = BayesianUncertaintyEstimator(num_samples=10)
        
        # Mock Bayesian model
        class SimpleBayesianModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 3)
                
            def forward(self, x):
                return self.linear(x) + torch.randn_like(self.linear(x)) * 0.1
        
        model = SimpleBayesianModel()
        input_data = torch.randn(16, 5)
        
        # Collect samples
        predictions = []
        for _ in range(10):
            with torch.no_grad():
                pred = model(input_data)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        result = estimator.estimate(predictions)
        
        assert "mean" in result
        assert "epistemic_uncertainty" in result
        assert result["mean"].shape == (16, 3)


class TestActiveLearningSelector:
    """Test suite for ActiveLearningSelector."""

    def setup_method(self):
        """Setup for each test."""
        self.selector = ActiveLearningSelector(strategy="uncertainty")
        
    def test_initialization(self):
        """Test active learning selector initialization."""
        assert self.selector.strategy == "uncertainty"
        
    def test_select_by_uncertainty(self):
        """Test selection by uncertainty."""
        uncertainties = torch.tensor([0.5, 2.0, 0.3, 1.5, 0.8])
        
        # Select top 2 most uncertain
        indices = self.selector.select(uncertainties, n=2)
        
        assert len(indices) == 2
        assert 1 in indices  # Index with uncertainty 2.0
        assert 3 in indices  # Index with uncertainty 1.5
        
    def test_select_by_diversity(self):
        """Test selection by diversity."""
        selector = ActiveLearningSelector(strategy="diversity")
        
        # Create embeddings
        embeddings = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [0.9, 0.9]
        ])
        
        indices = selector.select(embeddings, n=3)
        
        assert len(indices) == 3
        # Should select diverse points
        
    def test_select_by_information_gain(self):
        """Test selection by information gain."""
        selector = ActiveLearningSelector(strategy="information_gain")
        
        information_gains = torch.tensor([0.2, 0.8, 0.1, 0.9, 0.5])
        
        indices = selector.select(information_gains, n=2)
        
        assert len(indices) == 2
        assert 3 in indices  # Highest gain: 0.9
        assert 1 in indices  # Second highest: 0.8
        
    def test_select_more_than_available(self):
        """Test selecting more samples than available."""
        uncertainties = torch.tensor([0.5, 0.3, 0.8])
        
        # Request more than available
        indices = self.selector.select(uncertainties, n=10)
        
        # Should return all available
        assert len(indices) == 3
        
    def test_select_with_batch(self):
        """Test batch selection."""
        # Batch of uncertainties
        uncertainties = torch.rand(50)
        
        indices = self.selector.select(uncertainties, n=10)
        
        assert len(indices) == 10
        assert len(set(indices)) == 10  # No duplicates
        
    def test_random_strategy(self):
        """Test random selection strategy."""
        selector = ActiveLearningSelector(strategy="random")
        
        data = torch.rand(100)
        indices = selector.select(data, n=20)
        
        assert len(indices) == 20
        assert all(0 <= idx < 100 for idx in indices)
        
    def test_query_by_committee(self):
        """Test query by committee strategy."""
        selector = ActiveLearningSelector(strategy="committee")
        
        # Committee predictions: [num_models, batch_size, output_dim]
        committee_predictions = torch.randn(5, 30, 10)
        
        indices = selector.select(committee_predictions, n=5)
        
        assert len(indices) == 5


class TestUncertaintyIntegration:
    """Integration tests for uncertainty estimation."""

    def test_uncertainty_aware_decision_making(self):
        """Test using uncertainty for decision making."""
        from rtdi.world_models import ProbabilisticWorldModel
        
        model = ProbabilisticWorldModel(
            state_dim=8,
            action_dim=4,
            ensemble_size=3,
            hidden_dim=32
        )
        
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Make prediction
        state = torch.randn(1, 8)
        action = torch.randn(1, 4)
        
        prediction = model.forward(state, action)
        
        # Check uncertainty
        uncertainty = prediction["epistemic_uncertainty"]
        
        is_confident = estimator.is_high_confidence(
            torch.tensor([uncertainty])
        )
        
        assert isinstance(is_confident, torch.Tensor)
        
    def test_active_learning_with_uncertainty(self):
        """Test active learning pipeline with uncertainty."""
        from rtdi.world_models import ProbabilisticWorldModel
        
        model = ProbabilisticWorldModel(
            state_dim=6,
            action_dim=3,
            ensemble_size=3,
            hidden_dim=32
        )
        
        estimator = UncertaintyEstimator(method="ensemble")
        selector = ActiveLearningSelector(strategy="uncertainty")
        
        # Generate batch of states
        states = torch.randn(20, 6)
        actions = torch.randn(20, 3)
        
        # Compute uncertainties
        uncertainties = []
        for i in range(20):
            pred = model.forward(states[i:i+1], actions[i:i+1])
            uncertainties.append(pred["epistemic_uncertainty"])
        
        uncertainties = torch.tensor(uncertainties)
        
        # Select most uncertain samples
        selected_indices = selector.select(uncertainties, n=5)
        
        assert len(selected_indices) == 5
        
    def test_calibration_monitoring(self):
        """Test uncertainty calibration monitoring."""
        estimator = UncertaintyEstimator(method="ensemble")
        
        # Simulate predictions and errors over time
        predicted_stds = []
        actual_errors = []
        
        for _ in range(50):
            pred_std = torch.rand(1) * 2
            error = torch.rand(1) * 2
            
            predicted_stds.append(pred_std)
            actual_errors.append(error)
        
        predicted_stds = torch.cat(predicted_stds)
        actual_errors = torch.cat(actual_errors)
        
        calibration_error, scaling = estimator.calibrate_uncertainty(
            predicted_stds, actual_errors
        )
        
        # Should produce reasonable calibration metrics
        assert 0 <= calibration_error <= 10
        assert 0 < scaling < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])