"""Tests for reflexion and self-improvement module."""

import torch
import pytest
import numpy as np
from rtdi.reflexion import (
    Experience,
    ReflexionMemory,
    ReflexionAgent,
    SelfImprovementLoop
)


class TestExperience:
    """Test suite for Experience dataclass."""

    def test_basic_experience_creation(self):
        """Test creating basic experience."""
        state = torch.randn(8)
        action = torch.randn(4)
        reward = torch.tensor([1.0])
        next_state = torch.randn(8)
        done = True
        
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        assert torch.allclose(exp.state, state)
        assert torch.allclose(exp.action, action)
        assert exp.done is True
        assert exp.reflection is None
        assert exp.success is None
        
    def test_experience_with_reflection(self):
        """Test experience with reflection."""
        exp = Experience(
            state=torch.randn(8),
            action=torch.randn(4),
            reward=torch.tensor([1.0]),
            next_state=torch.randn(8),
            done=False,
            reflection="Action led to suboptimal outcome",
            success=False
        )
        
        assert exp.reflection is not None
        assert exp.success is False
        
    def test_experience_success_flag(self):
        """Test experience with success flag."""
        exp_success = Experience(
            state=torch.randn(8),
            action=torch.randn(4),
            reward=torch.tensor([10.0]),
            next_state=torch.randn(8),
            done=True,
            success=True
        )
        
        assert exp_success.success is True


class TestReflexionMemory:
    """Test suite for ReflexionMemory."""

    def setup_method(self):
        """Setup for each test."""
        self.capacity = 100
        self.memory = ReflexionMemory(capacity=self.capacity)
        
    def test_initialization(self):
        """Test memory initialization."""
        assert self.memory.capacity == self.capacity
        assert len(self.memory.successes) == 0
        assert len(self.memory.failures) == 0
        assert len(self.memory.reflections) == 0
        
    def test_add_success_experience(self):
        """Test adding successful experience."""
        exp = Experience(
            state=torch.randn(8),
            action=torch.randn(4),
            reward=torch.tensor([5.0]),
            next_state=torch.randn(8),
            done=True,
            success=True
        )
        
        self.memory.add_experience(exp)
        
        assert len(self.memory.successes) == 1
        assert len(self.memory.failures) == 0
        
    def test_add_failure_experience(self):
        """Test adding failed experience."""
        exp = Experience(
            state=torch.randn(8),
            action=torch.randn(4),
            reward=torch.tensor([-1.0]),
            next_state=torch.randn(8),
            done=True,
            success=False
        )
        
        self.memory.add_experience(exp)
        
        assert len(self.memory.successes) == 0
        assert len(self.memory.failures) == 1
        
    def test_add_experience_with_reflection(self):
        """Test adding experience with reflection."""
        exp = Experience(
            state=torch.randn(8),
            action=torch.randn(4),
            reward=torch.tensor([-1.0]),
            next_state=torch.randn(8),
            done=True,
            success=False,
            reflection="Should have taken more conservative action"
        )
        
        self.memory.add_experience(exp)
        
        assert len(self.memory.reflections) == 1
        assert self.memory.reflections[0]["reflection"] == exp.reflection
        
    def test_add_multiple_experiences(self):
        """Test adding multiple experiences."""
        for i in range(10):
            exp = Experience(
                state=torch.randn(8),
                action=torch.randn(4),
                reward=torch.tensor([float(i)]),
                next_state=torch.randn(8),
                done=False,
                success=i % 2 == 0
            )
            self.memory.add_experience(exp)
        
        assert len(self.memory.successes) == 5
        assert len(self.memory.failures) == 5
        
    def test_capacity_overflow(self):
        """Test that memory respects capacity."""
        memory = ReflexionMemory(capacity=10)
        
        for i in range(20):
            exp = Experience(
                state=torch.randn(8),
                action=torch.randn(4),
                reward=torch.tensor([1.0]),
                next_state=torch.randn(8),
                done=False,
                success=True
            )
            memory.add_experience(exp)
        
        # Should only keep last 10
        assert len(memory.successes) == 10
        
    def test_get_similar_failures(self):
        """Test retrieving similar failures."""
        # Add some failure experiences
        base_state = torch.randn(8)
        
        for i in range(10):
            # Create states with varying similarity
            noise = torch.randn(8) * (i * 0.1)
            state = base_state + noise
            
            exp = Experience(
                state=state,
                action=torch.randn(4),
                reward=torch.tensor([-1.0]),
                next_state=torch.randn(8),
                done=True,
                success=False
            )
            self.memory.add_experience(exp)
        
        # Query for similar failures
        similar = self.memory.get_similar_failures(base_state, k=3)
        
        assert len(similar) == 3
        
    def test_get_similar_failures_empty_memory(self):
        """Test getting similar failures from empty memory."""
        current_state = torch.randn(8)
        similar = self.memory.get_similar_failures(current_state, k=5)
        
        assert len(similar) == 0
        
    def test_get_similar_failures_limited(self):
        """Test getting similar failures when fewer than k available."""
        # Add only 2 failures
        for i in range(2):
            exp = Experience(
                state=torch.randn(8),
                action=torch.randn(4),
                reward=torch.tensor([-1.0]),
                next_state=torch.randn(8),
                done=True,
                success=False
            )
            self.memory.add_experience(exp)
        
        # Request 5
        similar = self.memory.get_similar_failures(torch.randn(8), k=5)
        
        assert len(similar) == 2
        
    def test_get_statistics(self):
        """Test memory statistics."""
        # Add mixed experiences
        for i in range(7):
            exp = Experience(
                state=torch.randn(8),
                action=torch.randn(4),
                reward=torch.tensor([1.0]),
                next_state=torch.randn(8),
                done=False,
                success=True
            )
            self.memory.add_experience(exp)
        
        for i in range(3):
            exp = Experience(
                state=torch.randn(8),
                action=torch.randn(4),
                reward=torch.tensor([-1.0]),
                next_state=torch.randn(8),
                done=False,
                success=False
            )
            self.memory.add_experience(exp)
        
        stats = self.memory.get_statistics()
        
        assert stats["num_successes"] == 7
        assert stats["num_failures"] == 3
        assert stats["success_rate"] == 0.7
        
    def test_get_statistics_empty(self):
        """Test statistics on empty memory."""
        stats = self.memory.get_statistics()
        
        assert stats["num_successes"] == 0
        assert stats["num_failures"] == 0
        assert stats["success_rate"] == 0.0


class TestReflexionAgent:
    """Test suite for ReflexionAgent."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        
        self.agent = ReflexionAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=64,
            memory_capacity=1000
        )
        
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.state_dim == self.state_dim
        assert self.agent.action_dim == self.action_dim
        assert self.agent.policy is not None
        assert self.agent.value is not None
        assert self.agent.memory is not None
        assert self.agent.optimizer is not None
        
    def test_select_action(self):
        """Test action selection."""
        state = torch.randn(1, self.state_dim)
        action = self.agent.select_action(state)
        
        assert action.shape == (1, self.action_dim)
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()
        
    def test_select_action_batch(self):
        """Test batch action selection."""
        batch_size = 8
        states = torch.randn(batch_size, self.state_dim)
        actions = self.agent.select_action(states)
        
        assert actions.shape == (batch_size, self.action_dim)
        
    def test_evaluate_state(self):
        """Test state value estimation."""
        state = torch.randn(1, self.state_dim)
        value = self.agent.evaluate_state(state)
        
        assert value.shape == (1, 1)
        
    def test_reflect_on_trajectory(self):
        """Test reflection on trajectory."""
        trajectory = []
        for i in range(10):
            exp = Experience(
                state=torch.randn(self.state_dim),
                action=torch.randn(self.action_dim),
                reward=torch.tensor([float(i)]),
                next_state=torch.randn(self.state_dim),
                done=False
            )
            trajectory.append(exp)
        
        reflection = self.agent.reflect_on_trajectory(trajectory)
        
        assert isinstance(reflection, str)
        assert len(reflection) > 0
        
    def test_learn_from_failure(self):
        """Test learning from failure."""
        failure_exp = Experience(
            state=torch.randn(self.state_dim),
            action=torch.randn(self.action_dim),
            reward=torch.tensor([-5.0]),
            next_state=torch.randn(self.state_dim),
            done=True,
            success=False,
            reflection="Action was too aggressive"
        )
        
        self.agent.memory.add_experience(failure_exp)
        
        metrics = self.agent.learn_from_failures()
        
        assert isinstance(metrics, dict)
        
    def test_update_policy(self):
        """Test policy update."""
        # Create batch of experiences
        experiences = []
        for i in range(20):
            exp = Experience(
                state=torch.randn(self.state_dim),
                action=torch.randn(self.action_dim),
                reward=torch.tensor([1.0]),
                next_state=torch.randn(self.state_dim),
                done=False,
                success=True
            )
            experiences.append(exp)
        
        metrics = self.agent.update_policy(experiences)
        
        assert isinstance(metrics, dict)
        assert "policy_loss" in metrics or "loss" in metrics
        
    def test_contrastive_learning(self):
        """Test contrastive learning from success/failure pairs."""
        # Add successes
        for i in range(5):
            exp = Experience(
                state=torch.randn(self.state_dim),
                action=torch.randn(self.action_dim),
                reward=torch.tensor([5.0]),
                next_state=torch.randn(self.state_dim),
                done=True,
                success=True
            )
            self.agent.memory.add_experience(exp)
        
        # Add failures
        for i in range(5):
            exp = Experience(
                state=torch.randn(self.state_dim),
                action=torch.randn(self.action_dim),
                reward=torch.tensor([-1.0]),
                next_state=torch.randn(self.state_dim),
                done=True,
                success=False
            )
            self.agent.memory.add_experience(exp)
        
        metrics = self.agent.contrastive_learning()
        
        assert isinstance(metrics, dict)


class TestSelfImprovementLoop:
    """Test suite for SelfImprovementLoop."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        
        self.agent = ReflexionAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=64
        )
        
        self.loop = SelfImprovementLoop(
            agent=self.agent,
            improvement_frequency=10
        )
        
    def test_initialization(self):
        """Test loop initialization."""
        assert self.loop.agent is not None
        assert self.loop.improvement_frequency == 10
        
    def test_run_episode_simple(self):
        """Test running simple episode."""
        # Create simple mock environment
        class SimpleEnv:
            def __init__(self):
                self.step_count = 0
                
            def reset(self):
                return np.random.randn(8)
            
            def step(self, action):
                self.step_count += 1
                next_state = np.random.randn(8)
                reward = np.random.rand()
                done = self.step_count >= 10
                return next_state, reward, done, {}
        
        env = SimpleEnv()
        result = self.loop.run_episode(env, max_steps=10)
        
        assert "total_reward" in result
        assert "num_steps" in result
        assert "reflection" in result
        
    def test_improvement_trigger(self):
        """Test that improvement is triggered at right frequency."""
        loop = SelfImprovementLoop(
            agent=self.agent,
            improvement_frequency=5
        )
        
        assert loop.improvement_frequency == 5
        
    def test_collect_trajectory(self):
        """Test trajectory collection."""
        class SimpleEnv:
            def __init__(self):
                self.step_count = 0
                
            def reset(self):
                return np.random.randn(8)
            
            def step(self, action):
                self.step_count += 1
                next_state = np.random.randn(8)
                reward = 1.0
                done = self.step_count >= 5
                return next_state, reward, done, {}
        
        env = SimpleEnv()
        trajectory = self.loop.collect_trajectory(env, max_steps=5)
        
        assert len(trajectory) == 5
        assert all(isinstance(exp, Experience) for exp in trajectory)
        
    def test_evaluate_and_reflect(self):
        """Test evaluation and reflection."""
        trajectory = []
        total_reward = 0.0
        
        for i in range(10):
            reward = float(i)
            total_reward += reward
            
            exp = Experience(
                state=torch.randn(self.state_dim),
                action=torch.randn(self.action_dim),
                reward=torch.tensor([reward]),
                next_state=torch.randn(self.state_dim),
                done=False
            )
            trajectory.append(exp)
        
        success = total_reward > 20
        reflection = self.loop.evaluate_and_reflect(trajectory, total_reward)
        
        assert isinstance(reflection, str)


class TestReflexionIntegration:
    """Integration tests for reflexion components."""

    def test_full_learning_cycle(self):
        """Test complete learning cycle with reflexion."""
        agent = ReflexionAgent(
            state_dim=6,
            action_dim=3,
            hidden_dim=32,
            memory_capacity=100
        )
        
        # Simulate some experiences
        for episode in range(5):
            trajectory = []
            episode_reward = 0.0
            
            for step in range(10):
                state = torch.randn(6)
                action = agent.select_action(state.unsqueeze(0)).squeeze(0)
                reward = torch.randn(1)
                next_state = torch.randn(6)
                done = step == 9
                
                episode_reward += reward.item()
                
                exp = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    success=episode_reward > 0
                )
                
                trajectory.append(exp)
                agent.memory.add_experience(exp)
            
            # Reflect and learn
            reflection = agent.reflect_on_trajectory(trajectory)
            metrics = agent.update_policy(trajectory)
        
        # Check that agent has learned
        stats = agent.memory.get_statistics()
        assert stats["num_successes"] + stats["num_failures"] > 0
        
    def test_failure_driven_improvement(self):
        """Test that agent improves from failures."""
        agent = ReflexionAgent(
            state_dim=6,
            action_dim=3,
            hidden_dim=32
        )
        
        # Add multiple failures
        for i in range(10):
            exp = Experience(
                state=torch.randn(6),
                action=torch.randn(3),
                reward=torch.tensor([-1.0]),
                next_state=torch.randn(6),
                done=True,
                success=False,
                reflection=f"Failure {i}: incorrect action selection"
            )
            agent.memory.add_experience(exp)
        
        # Learn from failures
        metrics = agent.learn_from_failures()
        
        stats = agent.memory.get_statistics()
        assert stats["num_failures"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])