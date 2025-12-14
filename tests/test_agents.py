"""Tests for agents."""

import torch
import pytest
from rtdi.agents import ModularAgent


class TestModularAgent:
    """Test suite for ModularAgent."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        
        self.agent = ModularAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config={
                "world_model": {"ensemble_size": 2, "hidden_dim": 32},
                "decision_loop": {"planning_horizon": 3},
                "learner": {"replay_capacity": 1000}
            }
        )

    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.state_dim == self.state_dim
        assert self.agent.action_dim == self.action_dim
        assert self.agent.world_model is not None
        assert self.agent.decision_loop is not None
        assert self.agent.learner is not None

    def test_act(self):
        """Test action selection."""
        state = torch.randn(1, self.state_dim)
        action = self.agent.act(state, time_budget=0.01)
        
        assert action.shape == (1, self.action_dim)

    def test_learn(self):
        """Test learning from experience."""
        experience = {
            "state": torch.randn(self.state_dim),
            "action": torch.randn(self.action_dim),
            "reward": torch.tensor([1.0]),
            "next_state": torch.randn(self.state_dim),
            "done": torch.tensor([False])
        }
        
        metrics = self.agent.learn(experience)
        
        # Should return some metrics (may be empty initially)
        assert isinstance(metrics, dict)

    def test_save_load(self):
        """Test saving and loading."""
        import tempfile
        import os
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "agent.pt")
            self.agent.save(save_path)
            
            # Verify file exists
            assert os.path.exists(save_path)
            
            # Create new agent and load
            new_agent = ModularAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim
            )
            new_agent.load(save_path)
            
            # Verify state is loaded
            assert new_agent.step_count == self.agent.step_count

    def test_get_stats(self):
        """Test getting statistics."""
        stats = self.agent.get_stats()
        
        assert "step_count" in stats
        assert "learner_stats" in stats
        assert isinstance(stats["step_count"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
