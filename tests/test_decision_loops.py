"""Tests for decision loops."""

import torch
import pytest
from rtdi.world_models import ProbabilisticWorldModel
from rtdi.decision_loops import (
    HierarchicalDecisionLoop,
    ReflexLoop,
    DeliberativeLoop,
    MetaLearningLoop,
    DecisionLevel
)


class TestReflexLoop:
    """Test suite for ReflexLoop."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 6
        self.action_dim = 3
        self.loop = ReflexLoop(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=32
        )

    def test_decide(self):
        """Test decision making."""
        state = torch.randn(1, self.state_dim)
        action = self.loop.decide(state, {})
        
        assert action.shape == (1, self.action_dim)
        assert (action >= -1).all() and (action <= 1).all()  # Tanh bounded

    def test_update(self):
        """Test update."""
        experience = {
            "states": torch.randn(10, self.state_dim),
            "actions": torch.randn(10, self.action_dim),
            "advantages": torch.randn(10)
        }
        
        metrics = self.loop.update(experience)
        
        assert "reflex_loss" in metrics
        assert metrics["reflex_loss"] >= 0


class TestDeliberativeLoop:
    """Test suite for DeliberativeLoop."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 6
        self.action_dim = 3
        
        # Create world model
        self.world_model = ProbabilisticWorldModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ensemble_size=2,
            hidden_dim=32
        )
        
        self.loop = DeliberativeLoop(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            world_model=self.world_model,
            planning_horizon=3,
            num_candidates=5
        )

    def test_decide(self):
        """Test planning-based decision."""
        state = torch.randn(1, self.state_dim)
        action = self.loop.decide(state, {})
        
        assert action.shape == (1, self.action_dim)


class TestMetaLearningLoop:
    """Test suite for MetaLearningLoop."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 6
        self.action_dim = 3
        self.loop = MetaLearningLoop(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            context_dim=16
        )

    def test_encode_task_context(self):
        """Test task context encoding."""
        trajectories = [
            {
                "state": torch.randn(self.state_dim),
                "action": torch.randn(self.action_dim),
                "reward": torch.tensor([1.0])
            }
            for _ in range(5)
        ]
        
        context = self.loop.encode_task_context(trajectories)
        
        assert context.shape == (1, self.loop.context_dim)

    def test_decide(self):
        """Test meta-level decision."""
        state = torch.randn(1, self.state_dim)
        trajectories = [
            {
                "state": torch.randn(self.state_dim),
                "action": torch.randn(self.action_dim),
                "reward": torch.tensor([0.5])
            }
        ]
        context = {"trajectories": trajectories}
        
        action = self.loop.decide(state, context)
        
        assert action.shape == (1, self.action_dim)


class TestHierarchicalDecisionLoop:
    """Test suite for HierarchicalDecisionLoop."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 6
        self.action_dim = 3
        
        self.world_model = ProbabilisticWorldModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ensemble_size=2,
            hidden_dim=32
        )
        
        self.loop = HierarchicalDecisionLoop(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            world_model=self.world_model,
            reflex_config={"hidden_dim": 16},
            deliberative_config={"planning_horizon": 3, "num_candidates": 5},
            meta_config={"context_dim": 8}
        )

    def test_decide_reflex(self):
        """Test reflex-level decision."""
        state = torch.randn(1, self.state_dim)
        action, level = self.loop.decide(state, time_budget=0.005)  # 5ms
        
        assert action.shape == (1, self.action_dim)
        assert level == DecisionLevel.REFLEX

    def test_decide_deliberative(self):
        """Test deliberative-level decision."""
        state = torch.randn(1, self.state_dim)
        action, level = self.loop.decide(state, time_budget=0.05)  # 50ms
        
        assert action.shape == (1, self.action_dim)
        assert level in [DecisionLevel.REFLEX, DecisionLevel.DELIBERATIVE]

    def test_decide_meta(self):
        """Test meta-level decision."""
        state = torch.randn(1, self.state_dim)
        trajectories = [
            {
                "state": torch.randn(self.state_dim),
                "action": torch.randn(self.action_dim),
                "reward": torch.tensor([0.5])
            }
        ]
        
        action, level = self.loop.decide(
            state,
            time_budget=0.5,
            context={"trajectories": trajectories}
        )
        
        assert action.shape == (1, self.action_dim)

    def test_update_all(self):
        """Test updating all loops."""
        experience = {
            "states": torch.randn(10, self.state_dim),
            "actions": torch.randn(10, self.action_dim),
            "advantages": torch.randn(10),
            "task_trajectories": []
        }
        
        metrics = self.loop.update_all(experience)
        
        assert "reflex_loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
