"""Tests for simulation and rollout module."""

import torch
import pytest
import numpy as np
from rtdi.simulation import (
    SimulationEngine,
    RolloutConfig,
    ModelBasedPlanner
)
from rtdi.world_models import ProbabilisticWorldModel


class TestRolloutConfig:
    """Test suite for RolloutConfig."""

    def test_default_initialization(self):
        """Test default config initialization."""
        config = RolloutConfig()
        
        assert config.horizon == 10
        assert config.num_samples == 10
        assert config.discount == 0.99
        assert config.temperature == 1.0
        assert config.use_mean is False
        
    def test_custom_initialization(self):
        """Test custom config initialization."""
        config = RolloutConfig(
            horizon=20,
            num_samples=5,
            discount=0.95,
            temperature=2.0,
            use_mean=True
        )
        
        assert config.horizon == 20
        assert config.num_samples == 5
        assert config.discount == 0.95
        assert config.temperature == 2.0
        assert config.use_mean is True


class TestSimulationEngine:
    """Test suite for SimulationEngine."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        
        self.world_model = ProbabilisticWorldModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ensemble_size=3,
            hidden_dim=64
        )
        
        self.engine = SimulationEngine(world_model=self.world_model)
        
    def test_initialization(self):
        """Test simulation engine initialization."""
        assert self.engine.world_model is not None
        assert self.engine.device is not None
        
    def test_initialization_with_custom_reward(self):
        """Test initialization with custom reward function."""
        def custom_reward(state, action, next_state):
            return -torch.norm(next_state, dim=-1, keepdim=True)
        
        engine = SimulationEngine(
            world_model=self.world_model,
            reward_function=custom_reward
        )
        
        assert engine.reward_function is not None
        
    def test_simple_rollout(self):
        """Test basic trajectory rollout."""
        initial_state = torch.randn(1, self.state_dim)
        
        def simple_policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=5, num_samples=1)
        
        trajectory = self.engine.rollout(initial_state, simple_policy, config)
        
        # Check output keys
        assert "states" in trajectory
        assert "actions" in trajectory
        assert "rewards" in trajectory
        assert "returns" in trajectory
        assert "uncertainties" in trajectory
        
    def test_rollout_shapes(self):
        """Test rollout output shapes."""
        batch_size = 4
        horizon = 10
        
        initial_state = torch.randn(batch_size, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=horizon)
        
        trajectory = self.engine.rollout(initial_state, policy, config)
        
        # Check shapes
        assert trajectory["states"].shape == (batch_size, horizon, self.state_dim)
        assert trajectory["actions"].shape == (batch_size, horizon, self.action_dim)
        assert trajectory["rewards"].shape == (batch_size, horizon, 1)
        assert trajectory["returns"].shape == (batch_size, horizon, 1)
        assert trajectory["uncertainties"].shape == (batch_size, horizon, 1)
        
    def test_rollout_with_mean(self):
        """Test rollout using mean predictions."""
        initial_state = torch.randn(2, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=5, use_mean=True)
        
        trajectory = self.engine.rollout(initial_state, policy, config)
        
        assert trajectory["states"].shape[0] == 2
        
    def test_rollout_with_sampling(self):
        """Test rollout with stochastic sampling."""
        initial_state = torch.randn(2, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=5, use_mean=False)
        
        trajectory = self.engine.rollout(initial_state, policy, config)
        
        assert trajectory["states"].shape[0] == 2
        
    def test_custom_reward_function(self):
        """Test rollout with custom reward function."""
        def custom_reward(state, action, next_state):
            # Reward based on state norm
            return -torch.norm(next_state, dim=-1, keepdim=True)
        
        engine = SimulationEngine(
            world_model=self.world_model,
            reward_function=custom_reward
        )
        
        initial_state = torch.randn(2, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=5)
        
        trajectory = engine.rollout(initial_state, policy, config)
        
        # Rewards should be negative (based on norm)
        assert (trajectory["rewards"] <= 0).all()
        
    def test_discount_calculation(self):
        """Test discounted returns calculation."""
        initial_state = torch.randn(1, self.state_dim)
        
        def policy(state):
            return torch.zeros(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=5, discount=0.9)
        
        trajectory = self.engine.rollout(initial_state, policy, config)
        
        # Check that returns are properly discounted
        rewards = trajectory["rewards"]
        returns = trajectory["returns"]
        
        assert returns.shape == rewards.shape
        # First return should be >= last return (due to discounting)
        assert returns[0, 0, 0] >= returns[0, -1, 0]
        
    def test_ensemble_rollout(self):
        """Test ensemble rollout for robustness."""
        initial_state = torch.randn(1, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=5, num_samples=3)
        
        rollout_data = self.engine.ensemble_rollout(initial_state, policy, config)
        
        # Should aggregate multiple rollouts
        assert "mean_return" in rollout_data or "states" in rollout_data
        
    def test_deterministic_policy_rollout(self):
        """Test rollout with deterministic policy."""
        initial_state = torch.randn(1, self.state_dim)
        
        def deterministic_policy(state):
            # Always return same action
            return torch.ones(state.shape[0], self.action_dim) * 0.5
        
        config = RolloutConfig(horizon=5)
        
        trajectory = self.engine.rollout(initial_state, deterministic_policy, config)
        
        # All actions should be the same
        actions = trajectory["actions"]
        expected_action = torch.ones(1, self.action_dim) * 0.5
        
        for t in range(actions.shape[1]):
            assert torch.allclose(actions[0, t], expected_action.squeeze(), atol=1e-5)
            
    def test_uncertainty_tracking(self):
        """Test that uncertainty is tracked during rollout."""
        initial_state = torch.randn(1, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=10)
        
        trajectory = self.engine.rollout(initial_state, policy, config)
        
        uncertainties = trajectory["uncertainties"]
        
        # Uncertainties should be non-negative
        assert (uncertainties >= 0).all()
        
    def test_long_horizon_rollout(self):
        """Test rollout with long horizon."""
        initial_state = torch.randn(1, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=50)
        
        trajectory = self.engine.rollout(initial_state, policy, config)
        
        assert trajectory["states"].shape[1] == 50
        
    def test_batch_rollout(self):
        """Test parallel rollout with batch."""
        batch_size = 8
        initial_state = torch.randn(batch_size, self.state_dim)
        
        def policy(state):
            return torch.randn(state.shape[0], self.action_dim)
        
        config = RolloutConfig(horizon=10)
        
        trajectory = self.engine.rollout(initial_state, policy, config)
        
        assert trajectory["states"].shape[0] == batch_size


class TestModelBasedPlanner:
    """Test suite for ModelBasedPlanner."""

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
        
        self.planner = ModelBasedPlanner(
            world_model=self.world_model,
            action_dim=self.action_dim,
            planning_horizon=5,
            num_candidates=10
        )
        
    def test_initialization(self):
        """Test planner initialization."""
        assert self.planner.world_model is not None
        assert self.planner.action_dim == self.action_dim
        assert self.planner.planning_horizon == 5
        assert self.planner.num_candidates == 10
        
    def test_plan_action(self):
        """Test planning single action."""
        state = torch.randn(1, self.state_dim)
        
        action = self.planner.plan(state)
        
        assert action.shape == (1, self.action_dim)
        
    def test_plan_with_different_methods(self):
        """Test planning with different optimization methods."""
        state = torch.randn(1, self.state_dim)
        
        # Random shooting
        planner_random = ModelBasedPlanner(
            world_model=self.world_model,
            action_dim=self.action_dim,
            planning_horizon=5,
            num_candidates=20,
            method="random_shooting"
        )
        
        action_random = planner_random.plan(state)
        assert action_random.shape == (1, self.action_dim)
        
        # CEM
        planner_cem = ModelBasedPlanner(
            world_model=self.world_model,
            action_dim=self.action_dim,
            planning_horizon=5,
            num_candidates=20,
            method="cem"
        )
        
        action_cem = planner_cem.plan(state)
        assert action_cem.shape == (1, self.action_dim)
        
    def test_plan_with_constraints(self):
        """Test planning with action constraints."""
        state = torch.randn(1, self.state_dim)
        
        planner = ModelBasedPlanner(
            world_model=self.world_model,
            action_dim=self.action_dim,
            planning_horizon=5,
            num_candidates=10,
            action_bounds=(-1.0, 1.0)
        )
        
        action = planner.plan(state)
        
        # Actions should be within bounds
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()
        
    def test_cem_iterations(self):
        """Test CEM planning iterations."""
        state = torch.randn(1, self.state_dim)
        
        planner = ModelBasedPlanner(
            world_model=self.world_model,
            action_dim=self.action_dim,
            planning_horizon=5,
            num_candidates=20,
            method="cem",
            num_iterations=5
        )
        
        action = planner.plan(state)
        
        assert action.shape == (1, self.action_dim)
        
    def test_plan_batch(self):
        """Test planning for batch of states."""
        batch_size = 4
        states = torch.randn(batch_size, self.state_dim)
        
        actions = self.planner.plan(states)
        
        assert actions.shape == (batch_size, self.action_dim)
        
    def test_get_action_sequence(self):
        """Test getting full action sequence."""
        state = torch.randn(1, self.state_dim)
        
        action_sequence = self.planner.get_action_sequence(state)
        
        # Should return full horizon of actions
        assert action_sequence.shape == (1, self.planner.planning_horizon, self.action_dim)
        
    def test_evaluate_trajectory(self):
        """Test trajectory evaluation."""
        state = torch.randn(1, self.state_dim)
        actions = torch.randn(1, 5, self.action_dim)  # horizon=5
        
        value = self.planner.evaluate_trajectory(state, actions)
        
        assert value.shape == (1,)
        
    def test_replan(self):
        """Test replanning after execution."""
        state = torch.randn(1, self.state_dim)
        
        # First plan
        action1 = self.planner.plan(state)
        
        # Simulate state change
        new_state = torch.randn(1, self.state_dim)
        
        # Replan
        action2 = self.planner.plan(new_state)
        
        assert action1.shape == action2.shape


class TestSimulationIntegration:
    """Integration tests for simulation components."""

    def test_plan_and_rollout(self):
        """Test planning followed by rollout."""
        state_dim = 6
        action_dim = 3
        
        world_model = ProbabilisticWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            ensemble_size=2,
            hidden_dim=32
        )
        
        planner = ModelBasedPlanner(
            world_model=world_model,
            action_dim=action_dim,
            planning_horizon=5,
            num_candidates=10
        )
        
        engine = SimulationEngine(world_model=world_model)
        
        # Plan action
        state = torch.randn(1, state_dim)
        action = planner.plan(state)
        
        # Rollout with planned action
        def planned_policy(s):
            return action.expand(s.shape[0], -1)
        
        config = RolloutConfig(horizon=5)
        trajectory = engine.rollout(state, planned_policy, config)
        
        assert trajectory["states"].shape[1] == 5
        
    def test_iterative_planning(self):
        """Test model-predictive control style planning."""
        state_dim = 6
        action_dim = 3
        
        world_model = ProbabilisticWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            ensemble_size=2,
            hidden_dim=32
        )
        
        planner = ModelBasedPlanner(
            world_model=world_model,
            action_dim=action_dim,
            planning_horizon=3,
            num_candidates=10
        )
        
        engine = SimulationEngine(world_model=world_model)
        
        state = torch.randn(1, state_dim)
        
        # MPC loop
        for step in range(5):
            # Plan
            action = planner.plan(state)
            
            # Simulate one step
            prediction = world_model.forward(state, action)
            state = prediction["next_state_mean"]
            
        assert state.shape == (1, state_dim)
        
    def test_uncertainty_aware_planning(self):
        """Test planning that considers uncertainty."""
        state_dim = 6
        action_dim = 3
        
        world_model = ProbabilisticWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            ensemble_size=3,
            hidden_dim=32
        )
        
        engine = SimulationEngine(world_model=world_model)
        
        state = torch.randn(1, state_dim)
        
        def policy(s):
            return torch.randn(s.shape[0], action_dim)
        
        config = RolloutConfig(horizon=10)
        trajectory = engine.rollout(state, policy, config)
        
        # Check uncertainty increases over horizon
        uncertainties = trajectory["uncertainties"]
        
        # Uncertainty should generally increase or stay positive
        assert (uncertainties >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])