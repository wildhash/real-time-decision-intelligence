"""Probabilistic World Models for uncertainty-aware state prediction."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.distributions as dist


class WorldModel(ABC):
    """Abstract base class for world models."""

    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict next state and rewards.
        
        Args:
            state: Current state tensor
            action: Action tensor
            
        Returns:
            Dictionary containing predictions and uncertainties
        """
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of training metrics
        """
        pass


class ProbabilisticWorldModel(WorldModel):
    """Probabilistic world model using ensemble and dropout for uncertainty estimation.
    
    This model predicts distributions over next states and rewards, providing
    both epistemic (model) and aleatoric (data) uncertainty estimates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        ensemble_size: int = 5,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        """Initialize probabilistic world model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            ensemble_size: Number of models in ensemble
            dropout_rate: Dropout rate for uncertainty
            learning_rate: Learning rate for optimizer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.ensemble_size = ensemble_size
        
        # Create ensemble of dynamics models
        self.ensemble = nn.ModuleList([
            self._build_dynamics_model(num_layers, dropout_rate)
            for _ in range(ensemble_size)
        ])
        
        # Optimizer for all models
        self.optimizer = torch.optim.Adam(
            self.ensemble.parameters(),
            lr=learning_rate
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble.to(self.device)

    def _build_dynamics_model(self, num_layers: int, dropout_rate: float) -> nn.Module:
        """Build a single dynamics model."""
        layers = []
        input_dim = self.state_dim + self.action_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output: mean and log_std for state delta, mean and log_std for reward
        output_dim = (self.state_dim + 1) * 2  # state_delta and reward, each with mean and log_std
        layers.append(nn.Linear(self.hidden_dim, output_dim))
        
        return nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Predict next state distribution using ensemble.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action to take [batch_size, action_dim]
            deterministic: If True, use mean prediction without sampling
            
        Returns:
            Dictionary with:
                - next_state_mean: Mean predicted next state
                - next_state_std: Uncertainty in next state
                - reward_mean: Mean predicted reward
                - reward_std: Uncertainty in reward
                - epistemic_uncertainty: Model uncertainty
                - aleatoric_uncertainty: Data uncertainty
        """
        state = state.to(self.device)
        action = action.to(self.device)
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Get predictions from all ensemble members
        ensemble_predictions = []
        for model in self.ensemble:
            pred = model(x)
            ensemble_predictions.append(pred)
        
        ensemble_predictions = torch.stack(ensemble_predictions, dim=0)  # [ensemble_size, batch, output_dim]
        
        # Split into state delta and reward predictions
        state_delta_mean = ensemble_predictions[:, :, :self.state_dim]
        state_delta_log_std = ensemble_predictions[:, :, self.state_dim:2*self.state_dim]
        reward_mean = ensemble_predictions[:, :, 2*self.state_dim:2*self.state_dim+1]
        reward_log_std = ensemble_predictions[:, :, -1:]
        
        # Compute epistemic uncertainty (variance across ensemble)
        state_epistemic = torch.var(state_delta_mean, dim=0)
        reward_epistemic = torch.var(reward_mean, dim=0)
        
        # Compute aleatoric uncertainty (average predicted variance)
        state_aleatoric = torch.mean(torch.exp(2 * state_delta_log_std), dim=0)
        reward_aleatoric = torch.mean(torch.exp(2 * reward_log_std), dim=0)
        
        # Mean predictions across ensemble
        state_delta_mean_avg = torch.mean(state_delta_mean, dim=0)
        reward_mean_avg = torch.mean(reward_mean, dim=0)
        
        # Next state prediction
        next_state_mean = state + state_delta_mean_avg
        
        # Total uncertainty (epistemic + aleatoric)
        next_state_std = torch.sqrt(state_epistemic + state_aleatoric)
        reward_std = torch.sqrt(reward_epistemic + reward_aleatoric)
        
        return {
            "next_state_mean": next_state_mean,
            "next_state_std": next_state_std,
            "reward_mean": reward_mean_avg,
            "reward_std": reward_std,
            "epistemic_uncertainty": state_epistemic.mean(),
            "aleatoric_uncertainty": state_aleatoric.mean(),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train the ensemble on a batch of transitions.
        
        Args:
            batch: Dictionary with keys 'states', 'actions', 'next_states', 'rewards'
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        
        # Compute state deltas
        state_deltas = next_states - states
        
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        losses = []
        
        # Train each ensemble member
        for model in self.ensemble:
            x = torch.cat([states, actions], dim=-1)
            pred = model(x)
            
            # Parse predictions
            state_delta_mean = pred[:, :self.state_dim]
            state_delta_log_std = pred[:, self.state_dim:2*self.state_dim]
            reward_mean = pred[:, 2*self.state_dim:2*self.state_dim+1]
            reward_log_std = pred[:, -1:]
            
            # Negative log-likelihood loss
            state_dist = dist.Normal(state_delta_mean, torch.exp(state_delta_log_std))
            reward_dist = dist.Normal(reward_mean, torch.exp(reward_log_std))
            
            state_loss = -state_dist.log_prob(state_deltas).mean()
            reward_loss = -reward_dist.log_prob(rewards.unsqueeze(-1)).mean()
            
            loss = state_loss + reward_loss
            losses.append(loss.item())
            total_loss += loss
        
        # Average loss across ensemble
        total_loss = total_loss / self.ensemble_size
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "state_loss": state_loss.item(),
            "reward_loss": reward_loss.item(),
        }

    def sample_trajectories(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """Sample trajectories from the probabilistic model.
        
        Args:
            initial_state: Initial state [batch_size, state_dim]
            actions: Sequence of actions [batch_size, horizon, action_dim]
            num_samples: Number of trajectory samples
            
        Returns:
            Dictionary with sampled states and rewards
        """
        batch_size, horizon, _ = actions.shape
        
        # Collect samples
        sampled_states = []
        sampled_rewards = []
        
        for _ in range(num_samples):
            states = [initial_state]
            rewards = []
            
            for t in range(horizon):
                pred = self.forward(states[-1], actions[:, t])
                
                # Sample next state
                next_state_dist = dist.Normal(
                    pred["next_state_mean"],
                    pred["next_state_std"]
                )
                next_state = next_state_dist.sample()
                
                # Sample reward
                reward_dist = dist.Normal(
                    pred["reward_mean"],
                    pred["reward_std"]
                )
                reward = reward_dist.sample()
                
                states.append(next_state)
                rewards.append(reward)
            
            sampled_states.append(torch.stack(states[1:], dim=1))  # [batch, horizon, state_dim]
            sampled_rewards.append(torch.stack(rewards, dim=1))  # [batch, horizon, 1]
        
        return {
            "states": torch.stack(sampled_states, dim=0),  # [num_samples, batch, horizon, state_dim]
            "rewards": torch.stack(sampled_rewards, dim=0),  # [num_samples, batch, horizon, 1]
        }
