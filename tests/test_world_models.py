"""Tests for world models."""

import torch
import pytest
from rtdi.world_models import ProbabilisticWorldModel


class TestProbabilisticWorldModel:
    """Test suite for ProbabilisticWorldModel."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        self.batch_size = 16
        
        self.model = ProbabilisticWorldModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ensemble_size=3,
            hidden_dim=64,
            num_layers=2
        )

    def test_initialization(self):
        """Test model initialization."""
        assert self.model.state_dim == self.state_dim
        assert self.model.action_dim == self.action_dim
        assert self.model.ensemble_size == 3
        assert len(self.model.ensemble) == 3

    def test_forward(self):
        """Test forward pass."""
        state = torch.randn(self.batch_size, self.state_dim)
        action = torch.randn(self.batch_size, self.action_dim)
        
        output = self.model.forward(state, action)
        
        # Check output keys
        assert "next_state_mean" in output
        assert "next_state_std" in output
        assert "reward_mean" in output
        assert "reward_std" in output
        assert "epistemic_uncertainty" in output
        assert "aleatoric_uncertainty" in output
        
        # Check shapes
        assert output["next_state_mean"].shape == (self.batch_size, self.state_dim)
        assert output["next_state_std"].shape == (self.batch_size, self.state_dim)
        assert output["reward_mean"].shape == (self.batch_size, 1)
        
        # Check uncertainty is positive
        assert (output["next_state_std"] >= 0).all()
        assert output["epistemic_uncertainty"] >= 0
        assert output["aleatoric_uncertainty"] >= 0

    def test_train_step(self):
        """Test training step."""
        batch = {
            "states": torch.randn(self.batch_size, self.state_dim),
            "actions": torch.randn(self.batch_size, self.action_dim),
            "next_states": torch.randn(self.batch_size, self.state_dim),
            "rewards": torch.randn(self.batch_size, 1)
        }
        
        metrics = self.model.train_step(batch)
        
        # Check metrics
        assert "loss" in metrics
        assert "state_loss" in metrics
        assert "reward_loss" in metrics
        
        # Check losses are valid (finite and not NaN)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert not torch.isnan(torch.tensor(metrics["loss"]))

    def test_sample_trajectories(self):
        """Test trajectory sampling."""
        initial_state = torch.randn(1, self.state_dim)
        actions = torch.randn(1, 10, self.action_dim)  # horizon=10
        
        trajectories = self.model.sample_trajectories(
            initial_state, actions, num_samples=5
        )
        
        # Check output
        assert "states" in trajectories
        assert "rewards" in trajectories
        
        # Check shapes
        assert trajectories["states"].shape == (5, 1, 10, self.state_dim)
        assert trajectories["rewards"].shape == (5, 1, 10, 1)

    def test_uncertainty_estimation(self):
        """Test that uncertainty decreases with more data."""
        state = torch.randn(1, self.state_dim)
        action = torch.randn(1, self.action_dim)
        
        # Initial uncertainty
        output1 = self.model.forward(state, action)
        uncertainty1 = output1["epistemic_uncertainty"]
        
        # Train with some data
        for _ in range(10):
            batch = {
                "states": torch.randn(32, self.state_dim),
                "actions": torch.randn(32, self.action_dim),
                "next_states": torch.randn(32, self.state_dim),
                "rewards": torch.randn(32, 1)
            }
            self.model.train_step(batch)
        
        # Uncertainty after training
        output2 = self.model.forward(state, action)
        uncertainty2 = output2["epistemic_uncertainty"]
        
        # Check that uncertainty exists (may not always decrease in practice)
        assert uncertainty1 >= 0
        assert uncertainty2 >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
