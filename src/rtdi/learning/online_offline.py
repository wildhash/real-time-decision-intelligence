"""Online and Offline Learning modules."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from collections import deque
import numpy as np


class ReplayBuffer:
    """Experience replay buffer for offline learning."""

    def __init__(self, capacity: int = 100000, prioritized: bool = False):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            prioritized: Whether to use prioritized sampling
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity) if prioritized else None
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4  # Importance sampling exponent
        self.epsilon = 1e-6  # Small constant for numerical stability

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        priority: Optional[float] = None
    ):
        """Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            priority: Optional priority for prioritized replay
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        if self.prioritized:
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority if priority is not None else max_priority)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample batch from buffer.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Tuple of (batch_dict, weights, indices)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.prioritized:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.FloatTensor(weights)
            
            samples = [self.buffer[idx] for idx in indices]
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[idx] for idx in indices]
            weights = None
        
        # Stack samples
        states = torch.stack([s[0] for s in samples])
        actions = torch.stack([s[1] for s in samples])
        rewards = torch.stack([s[2] for s in samples])
        next_states = torch.stack([s[3] for s in samples])
        dones = torch.stack([s[4] for s in samples])
        
        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }
        
        return batch, weights, torch.LongTensor(indices) if self.prioritized else None

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for prioritized replay.
        
        Args:
            indices: Indices of samples
            priorities: New priorities
        """
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx.item()] = priority.item() + self.epsilon

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineLearner:
    """Online learning from streaming data."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        buffer_size: int = 1000,
        update_frequency: int = 1
    ):
        """Initialize online learner.
        
        Args:
            model: Model to train online
            learning_rate: Learning rate
            buffer_size: Size of online buffer
            update_frequency: Steps between updates
        """
        self.model = model
        
        # Only create optimizer if model has parameters
        if hasattr(model, 'parameters') and callable(model.parameters):
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = None
            
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.update_frequency = update_frequency
        self.step_count = 0

    def observe(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ):
        """Observe new experience and update online.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # Add to buffer
        self.buffer.add(state, action, reward, next_state, done)
        
        self.step_count += 1
        
        # Update if frequency met
        if self.step_count % self.update_frequency == 0 and len(self.buffer) > 0:
            return self.update()
        
        return {}

    def update(self) -> Dict[str, float]:
        """Perform online update.
        
        Returns:
            Update metrics
        """
        # Sample recent experiences
        batch, _, _ = self.buffer.sample(min(32, len(self.buffer)))
        
        # Update model (implementation depends on model type)
        # For world model:
        if hasattr(self.model, 'train_step'):
            metrics = self.model.train_step(batch)
        else:
            # Return default metrics if train_step is unavailable
            metrics = {"update_performed": False, "loss": float('nan')}
        
        return metrics


class OfflineLearner:
    """Offline learning from fixed dataset."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        num_epochs: int = 100
    ):
        """Initialize offline learner.
        
        Args:
            model: Model to train offline
            learning_rate: Learning rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
        """
        self.model = model
        
        # Only create optimizer if model has parameters
        if hasattr(model, 'parameters') and callable(model.parameters):
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = None
            
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(
        self,
        dataset: ReplayBuffer,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """Train on offline dataset.
        
        Args:
            dataset: Replay buffer with offline data
            validation_split: Fraction of data for validation
            
        Returns:
            Training history
        """
        history = {"train_loss": [], "val_loss": []}
        
        # Split dataset
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        # Note: This is a simplified implementation. Both train and validation
        # sample from the same buffer. For proper validation, consider 
        # partitioning the buffer indices or using held-out data.
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_losses = []
            for _ in range(train_size // self.batch_size):
                batch, _, _ = dataset.sample(self.batch_size)
                
                if hasattr(self.model, 'train_step'):
                    metrics = self.model.train_step(batch)
                    train_losses.append(metrics.get('loss', 0.0))
            
            # Validation phase
            val_losses = []
            for _ in range(val_size // self.batch_size):
                batch, _, _ = dataset.sample(self.batch_size)
                
                with torch.no_grad():
                    if hasattr(self.model, 'train_step'):
                        # Compute validation loss without updates
                        pred = self.model.forward(batch["states"], batch["actions"])
                        # Simple MSE for validation
                        val_loss = ((pred["next_state_mean"] - batch["next_states"]) ** 2).mean()
                        val_losses.append(val_loss.item())
            
            if train_losses:
                history["train_loss"].append(np.mean(train_losses))
            if val_losses:
                history["val_loss"].append(np.mean(val_losses))
        
        return history


class HybridLearner:
    """Hybrid online + offline learning system."""

    def __init__(
        self,
        model: nn.Module,
        online_config: Optional[Dict[str, Any]] = None,
        offline_config: Optional[Dict[str, Any]] = None,
        replay_capacity: int = 100000
    ):
        """Initialize hybrid learner.
        
        Args:
            model: Model to train
            online_config: Configuration for online learning
            offline_config: Configuration for offline learning
            replay_capacity: Capacity of replay buffer
        """
        self.model = model
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, prioritized=True)
        
        # Create online and offline learners with proper config
        online_cfg = online_config or {}
        offline_cfg = offline_config or {}
        
        # Don't pass model to sub-learners if it doesn't have parameters attribute
        if hasattr(model, 'parameters'):
            self.online_learner = OnlineLearner(model, **online_cfg)
            self.offline_learner = OfflineLearner(model, **offline_cfg)
        else:
            # For models that don't inherit from nn.Module (like ProbabilisticWorldModel)
            self.online_learner = OnlineLearner(model, **online_cfg)
            self.offline_learner = OfflineLearner(model, **offline_cfg)
        
        self.online_steps = 0
        self.offline_updates = 0

    def observe_online(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ) -> Dict[str, float]:
        """Process online experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            
        Returns:
            Update metrics
        """
        # Add to main replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Online update
        metrics = self.online_learner.observe(state, action, reward, next_state, done)
        self.online_steps += 1
        
        return metrics

    def train_offline(
        self,
        num_epochs: int = 10
    ) -> Dict[str, List[float]]:
        """Perform offline training on collected data.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Training history
        """
        self.offline_learner.num_epochs = num_epochs
        history = self.offline_learner.train(self.replay_buffer)
        self.offline_updates += 1
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "online_steps": self.online_steps,
            "offline_updates": self.offline_updates,
            "buffer_size": len(self.replay_buffer),
            "buffer_capacity": self.replay_buffer.capacity,
        }
