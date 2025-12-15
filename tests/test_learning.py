"""Tests for learning module."""

import torch
import pytest
import numpy as np
from rtdi.learning import (
    ReplayBuffer,
    OnlineLearner,
    OfflineLearner,
    HybridLearner
)
from rtdi.world_models import ProbabilisticWorldModel


class TestReplayBuffer:
    """Test suite for ReplayBuffer."""

    def setup_method(self):
        """Setup for each test."""
        self.capacity = 100
        self.state_dim = 8
        self.action_dim = 4
        
    def test_initialization_standard(self):
        """Test standard buffer initialization."""
        buffer = ReplayBuffer(capacity=self.capacity, prioritized=False)
        
        assert buffer.capacity == self.capacity
        assert buffer.prioritized is False
        assert len(buffer) == 0
        assert buffer.priorities is None
        
    def test_initialization_prioritized(self):
        """Test prioritized buffer initialization."""
        buffer = ReplayBuffer(capacity=self.capacity, prioritized=True)
        
        assert buffer.capacity == self.capacity
        assert buffer.prioritized is True
        assert len(buffer) == 0
        assert buffer.priorities is not None
        
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        buffer = ReplayBuffer(capacity=self.capacity)
        
        state = torch.randn(self.state_dim)
        action = torch.randn(self.action_dim)
        reward = torch.tensor([1.0])
        next_state = torch.randn(self.state_dim)
        done = torch.tensor([False])
        
        buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
        
    def test_add_multiple_experiences(self):
        """Test adding multiple experiences."""
        buffer = ReplayBuffer(capacity=self.capacity)
        
        for i in range(50):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([float(i)])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 50
        
    def test_capacity_overflow(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        # Add more than capacity
        for i in range(20):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([float(i)])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            buffer.add(state, action, reward, next_state, done)
        
        # Should only keep last 10
        assert len(buffer) == 10
        
    def test_sample_standard(self):
        """Test sampling from standard buffer."""
        buffer = ReplayBuffer(capacity=self.capacity)
        
        # Add experiences
        for i in range(50):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([float(i)])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            buffer.add(state, action, reward, next_state, done)
        
        # Sample batch
        batch, weights, indices = buffer.sample(batch_size=16)
        
        assert batch["states"].shape == (16, self.state_dim)
        assert batch["actions"].shape == (16, self.action_dim)
        assert batch["rewards"].shape == (16, 1)
        assert batch["next_states"].shape == (16, self.state_dim)
        assert batch["dones"].shape == (16, 1)
        assert weights is None  # Standard buffer doesn't use weights
        
    def test_sample_prioritized(self):
        """Test sampling from prioritized buffer."""
        buffer = ReplayBuffer(capacity=self.capacity, prioritized=True)
        
        # Add experiences with priorities
        for i in range(50):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([float(i)])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            priority = float(i + 1)
            
            buffer.add(state, action, reward, next_state, done, priority=priority)
        
        # Sample batch
        batch, weights, indices = buffer.sample(batch_size=16)
        
        assert batch["states"].shape == (16, self.state_dim)
        assert weights is not None
        assert weights.shape == (16,)
        assert indices is not None
        assert indices.shape == (16,)
        assert (weights > 0).all()
        assert (weights <= 1.0).all()
        
    def test_update_priorities(self):
        """Test updating priorities in prioritized buffer."""
        buffer = ReplayBuffer(capacity=self.capacity, prioritized=True)
        
        # Add experiences
        for i in range(20):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            buffer.add(state, action, reward, next_state, done)
        
        # Sample and update priorities
        batch, weights, indices = buffer.sample(batch_size=10)
        new_priorities = torch.rand(10) * 10
        
        buffer.update_priorities(indices, new_priorities)
        
        # Verify priorities were updated
        for idx, priority in zip(indices, new_priorities):
            assert buffer.priorities[idx.item()] > 0
            
    def test_sample_small_buffer(self):
        """Test sampling when buffer has fewer items than batch size."""
        buffer = ReplayBuffer(capacity=self.capacity)
        
        # Add only 5 experiences
        for i in range(5):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            buffer.add(state, action, reward, next_state, done)
        
        # Request larger batch
        batch, weights, indices = buffer.sample(batch_size=10)
        
        # Should return all 5 items
        assert batch["states"].shape[0] == 5


class TestOnlineLearner:
    """Test suite for OnlineLearner."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        
        self.model = ProbabilisticWorldModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ensemble_size=2,
            hidden_dim=32
        )
        
        self.learner = OnlineLearner(
            model=self.model,
            learning_rate=1e-3,
            buffer_size=1000,
            update_frequency=1
        )
        
    def test_initialization(self):
        """Test online learner initialization."""
        assert self.learner.model is not None
        assert self.learner.optimizer is not None
        assert len(self.learner.buffer) == 0
        assert self.learner.step_count == 0
        
    def test_observe_single_experience(self):
        """Test observing single experience."""
        state = torch.randn(self.state_dim)
        action = torch.randn(self.action_dim)
        reward = torch.tensor([1.0])
        next_state = torch.randn(self.state_dim)
        done = torch.tensor([False])
        
        metrics = self.learner.observe(state, action, reward, next_state, done)
        
        assert isinstance(metrics, dict)
        assert len(self.learner.buffer) == 1
        assert self.learner.step_count == 1
        
    def test_observe_multiple_experiences(self):
        """Test observing multiple experiences."""
        for i in range(10):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([float(i)])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            self.learner.observe(state, action, reward, next_state, done)
        
        assert len(self.learner.buffer) == 10
        assert self.learner.step_count == 10
        
    def test_learning_updates(self):
        """Test that learning updates occur."""
        # Add enough experiences to trigger updates
        for i in range(20):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            metrics = self.learner.observe(state, action, reward, next_state, done)
        
        # Should have performed updates
        assert self.learner.step_count == 20
        
    def test_get_stats(self):
        """Test getting learner statistics."""
        stats = self.learner.get_stats()
        
        assert "buffer_size" in stats
        assert "step_count" in stats
        assert stats["buffer_size"] == len(self.learner.buffer)
        assert stats["step_count"] == self.learner.step_count


class TestOfflineLearner:
    """Test suite for OfflineLearner."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        
        self.model = ProbabilisticWorldModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ensemble_size=2,
            hidden_dim=32
        )
        
        self.learner = OfflineLearner(
            model=self.model,
            batch_size=16,
            num_epochs=5
        )
        
    def test_initialization(self):
        """Test offline learner initialization."""
        assert self.learner.model is not None
        assert self.learner.optimizer is not None
        assert self.learner.batch_size == 16
        assert self.learner.num_epochs == 5
        
    def test_train_with_dataset(self):
        """Test training with dataset."""
        # Create dataset
        dataset = ReplayBuffer(capacity=1000)
        
        for i in range(100):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            dataset.add(state, action, reward, next_state, done)
        
        # Train
        history = self.learner.train(dataset, num_epochs=3)
        
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
        assert all(loss > 0 for loss in history["train_loss"])
        
    def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        dataset = ReplayBuffer(capacity=100)
        
        history = self.learner.train(dataset, num_epochs=1)
        
        # Should handle gracefully
        assert isinstance(history, dict)
        
    def test_validation_split(self):
        """Test training with validation split."""
        # Create dataset
        dataset = ReplayBuffer(capacity=1000)
        
        for i in range(100):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            dataset.add(state, action, reward, next_state, done)
        
        # Train with validation
        history = self.learner.train(dataset, num_epochs=2, validation_split=0.2)
        
        assert "train_loss" in history
        # May or may not have val_loss depending on implementation


class TestHybridLearner:
    """Test suite for HybridLearner."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.action_dim = 4
        
        self.model = ProbabilisticWorldModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ensemble_size=2,
            hidden_dim=32
        )
        
        self.learner = HybridLearner(
            model=self.model,
            online_learning_rate=1e-3,
            replay_capacity=1000,
            batch_size=16
        )
        
    def test_initialization(self):
        """Test hybrid learner initialization."""
        assert self.learner.model is not None
        assert self.learner.online_learner is not None
        assert self.learner.offline_learner is not None
        assert self.learner.replay_buffer is not None
        
    def test_online_observation(self):
        """Test online learning component."""
        state = torch.randn(self.state_dim)
        action = torch.randn(self.action_dim)
        reward = torch.tensor([1.0])
        next_state = torch.randn(self.state_dim)
        done = torch.tensor([False])
        
        metrics = self.learner.observe_online(state, action, reward, next_state, done)
        
        assert isinstance(metrics, dict)
        assert len(self.learner.replay_buffer) == 1
        
    def test_offline_training(self):
        """Test offline training component."""
        # Populate replay buffer
        for i in range(50):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            self.learner.observe_online(state, action, reward, next_state, done)
        
        # Perform offline training
        history = self.learner.train_offline(num_epochs=2)
        
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
        
    def test_hybrid_learning_cycle(self):
        """Test complete hybrid learning cycle."""
        # Online phase
        for i in range(30):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            self.learner.observe_online(state, action, reward, next_state, done)
        
        # Offline phase
        history = self.learner.train_offline(num_epochs=2)
        
        # Continue online learning
        for i in range(10):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            metrics = self.learner.observe_online(state, action, reward, next_state, done)
        
        assert len(self.learner.replay_buffer) == 40
        
    def test_get_stats(self):
        """Test getting hybrid learner statistics."""
        stats = self.learner.get_stats()
        
        assert "buffer_size" in stats
        assert "online_steps" in stats
        assert isinstance(stats, dict)
        
    def test_prioritized_replay(self):
        """Test with prioritized replay buffer."""
        learner = HybridLearner(
            model=self.model,
            online_learning_rate=1e-3,
            replay_capacity=1000,
            prioritized_replay=True
        )
        
        # Add experiences
        for i in range(20):
            state = torch.randn(self.state_dim)
            action = torch.randn(self.action_dim)
            reward = torch.tensor([1.0])
            next_state = torch.randn(self.state_dim)
            done = torch.tensor([False])
            
            learner.observe_online(state, action, reward, next_state, done)
        
        assert learner.replay_buffer.prioritized is True
        assert len(learner.replay_buffer) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])