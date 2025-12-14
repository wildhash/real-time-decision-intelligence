"""Simulation and rollout engine for model-based planning."""

from typing import Dict, Any, List, Optional, Callable
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class RolloutConfig:
    """Configuration for simulation rollouts."""
    horizon: int = 10
    num_samples: int = 10
    discount: float = 0.99
    temperature: float = 1.0
    use_mean: bool = False


class SimulationEngine:
    """Engine for simulating trajectories using world models."""

    def __init__(
        self,
        world_model: Any,
        reward_function: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize simulation engine.
        
        Args:
            world_model: World model for predictions
            reward_function: Optional custom reward function
            device: Torch device
        """
        self.world_model = world_model
        self.reward_function = reward_function
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rollout(
        self,
        initial_state: torch.Tensor,
        policy: Callable[[torch.Tensor], torch.Tensor],
        config: RolloutConfig
    ) -> Dict[str, torch.Tensor]:
        """Rollout policy in simulated environment.
        
        Args:
            initial_state: Starting state [batch_size, state_dim]
            policy: Policy function mapping states to actions
            config: Rollout configuration
            
        Returns:
            Dictionary with trajectory data
        """
        batch_size = initial_state.shape[0]
        
        # Initialize trajectory storage
        states = [initial_state]
        actions = []
        rewards = []
        uncertainties = []
        
        current_state = initial_state
        
        for t in range(config.horizon):
            # Get action from policy
            action = policy(current_state)
            actions.append(action)
            
            # Predict next state using world model
            prediction = self.world_model.forward(current_state, action)
            
            if config.use_mean:
                next_state = prediction["next_state_mean"]
            else:
                # Sample from predictive distribution
                next_state = prediction["next_state_mean"] + \
                            torch.randn_like(prediction["next_state_mean"]) * prediction["next_state_std"]
            
            # Get reward
            if self.reward_function is not None:
                reward = self.reward_function(current_state, action, next_state)
            else:
                reward = prediction["reward_mean"]
            
            rewards.append(reward)
            states.append(next_state)
            
            # Track uncertainty
            uncertainty = prediction["next_state_std"].mean(dim=-1, keepdim=True)
            uncertainties.append(uncertainty)
            
            current_state = next_state
        
        # Stack trajectories
        states = torch.stack(states[:-1], dim=1)  # [batch, horizon, state_dim]
        actions = torch.stack(actions, dim=1)  # [batch, horizon, action_dim]
        rewards = torch.stack(rewards, dim=1)  # [batch, horizon, 1]
        uncertainties = torch.stack(uncertainties, dim=1)  # [batch, horizon, 1]
        
        # Compute returns
        returns = self._compute_returns(rewards, config.discount)
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns": returns,
            "uncertainties": uncertainties,
        }

    def _compute_returns(self, rewards: torch.Tensor, discount: float) -> torch.Tensor:
        """Compute discounted returns.
        
        Args:
            rewards: Reward tensor [batch, horizon, 1]
            discount: Discount factor
            
        Returns:
            Returns tensor [batch, horizon, 1]
        """
        batch_size, horizon, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        
        running_return = torch.zeros(batch_size, 1).to(rewards.device)
        for t in reversed(range(horizon)):
            running_return = rewards[:, t] + discount * running_return
            returns[:, t] = running_return
        
        return returns

    def ensemble_rollout(
        self,
        initial_state: torch.Tensor,
        policy: Callable[[torch.Tensor], torch.Tensor],
        config: RolloutConfig
    ) -> Dict[str, torch.Tensor]:
        """Perform multiple rollouts and aggregate.
        
        Args:
            initial_state: Starting state [batch_size, state_dim]
            policy: Policy function
            config: Rollout configuration
            
        Returns:
            Aggregated rollout data
        """
        rollouts = []
        
        for _ in range(config.num_samples):
            rollout = self.rollout(initial_state, policy, config)
            rollouts.append(rollout)
        
        # Aggregate rollouts
        aggregated = {
            "states_mean": torch.stack([r["states"] for r in rollouts]).mean(dim=0),
            "states_std": torch.stack([r["states"] for r in rollouts]).std(dim=0),
            "rewards_mean": torch.stack([r["rewards"] for r in rollouts]).mean(dim=0),
            "rewards_std": torch.stack([r["rewards"] for r in rollouts]).std(dim=0),
            "returns_mean": torch.stack([r["returns"] for r in rollouts]).mean(dim=0),
            "returns_std": torch.stack([r["returns"] for r in rollouts]).std(dim=0),
            "uncertainties": torch.stack([r["uncertainties"] for r in rollouts]).mean(dim=0),
        }
        
        return aggregated

    def evaluate_policy(
        self,
        policy: Callable[[torch.Tensor], torch.Tensor],
        num_episodes: int,
        initial_states: torch.Tensor,
        config: RolloutConfig
    ) -> Dict[str, float]:
        """Evaluate policy performance.
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of evaluation episodes
            initial_states: Initial states for episodes
            config: Rollout configuration
            
        Returns:
            Evaluation metrics
        """
        total_return = 0.0
        total_uncertainty = 0.0
        
        for i in range(num_episodes):
            # Get initial state
            init_state = initial_states[i:i+1]
            
            # Rollout
            rollout = self.rollout(init_state, policy, config)
            
            total_return += rollout["returns"][:, 0].mean().item()
            total_uncertainty += rollout["uncertainties"].mean().item()
        
        return {
            "mean_return": total_return / num_episodes,
            "mean_uncertainty": total_uncertainty / num_episodes,
        }


class ModelBasedPlanner:
    """Model-based planner using cross-entropy method or shooting."""

    def __init__(
        self,
        simulation_engine: SimulationEngine,
        method: str = "cem",
        num_iterations: int = 5,
        num_candidates: int = 100,
        num_elite: int = 10
    ):
        """Initialize model-based planner.
        
        Args:
            simulation_engine: Simulation engine
            method: Planning method ('cem', 'random_shooting')
            num_iterations: Number of optimization iterations
            num_candidates: Number of action sequences to sample
            num_elite: Number of elite sequences for CEM
        """
        self.simulation_engine = simulation_engine
        self.method = method
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elite = num_elite

    def plan(
        self,
        initial_state: torch.Tensor,
        horizon: int,
        action_dim: int
    ) -> torch.Tensor:
        """Plan action sequence.
        
        Args:
            initial_state: Starting state [1, state_dim]
            horizon: Planning horizon
            action_dim: Action dimension
            
        Returns:
            Planned action sequence [horizon, action_dim]
        """
        if self.method == "cem":
            return self._cem_planning(initial_state, horizon, action_dim)
        elif self.method == "random_shooting":
            return self._random_shooting(initial_state, horizon, action_dim)
        else:
            raise ValueError(f"Unknown planning method: {self.method}")

    def _random_shooting(
        self,
        initial_state: torch.Tensor,
        horizon: int,
        action_dim: int
    ) -> torch.Tensor:
        """Random shooting planning.
        
        Args:
            initial_state: Starting state
            horizon: Planning horizon
            action_dim: Action dimension
            
        Returns:
            Best action sequence
        """
        device = initial_state.device
        
        # Sample random action sequences
        action_sequences = torch.randn(
            self.num_candidates, horizon, action_dim
        ).to(device)
        action_sequences = torch.tanh(action_sequences)  # Bound actions
        
        # Evaluate each sequence
        best_return = float('-inf')
        best_sequence = None
        
        for i in range(self.num_candidates):
            actions = action_sequences[i:i+1]  # [1, horizon, action_dim]
            
            # Expand initial state
            state = initial_state.repeat(1, 1)
            
            # Simple policy that follows action sequence
            def fixed_policy(s, t=[0]):
                if t[0] < actions.shape[1]:
                    action = actions[:, t[0]]
                    t[0] += 1
                    return action
                return torch.zeros(1, action_dim).to(device)
            
            # Rollout
            config = RolloutConfig(horizon=horizon, num_samples=1)
            rollout = self.simulation_engine.rollout(state, fixed_policy, config)
            
            # Compute return
            total_return = rollout["returns"][:, 0].item()
            
            if total_return > best_return:
                best_return = total_return
                best_sequence = action_sequences[i]
        
        return best_sequence

    def _cem_planning(
        self,
        initial_state: torch.Tensor,
        horizon: int,
        action_dim: int
    ) -> torch.Tensor:
        """Cross-entropy method planning.
        
        Args:
            initial_state: Starting state
            horizon: Planning horizon
            action_dim: Action dimension
            
        Returns:
            Best action sequence
        """
        device = initial_state.device
        
        # Initialize distribution
        mean = torch.zeros(horizon, action_dim).to(device)
        std = torch.ones(horizon, action_dim).to(device)
        
        for iteration in range(self.num_iterations):
            # Sample action sequences from current distribution
            action_sequences = mean.unsqueeze(0) + \
                             std.unsqueeze(0) * torch.randn(
                                 self.num_candidates, horizon, action_dim
                             ).to(device)
            action_sequences = torch.tanh(action_sequences)
            
            # Evaluate sequences
            returns = []
            for i in range(self.num_candidates):
                actions = action_sequences[i:i+1]
                
                state = initial_state
                total_return = 0.0
                
                for t in range(horizon):
                    pred = self.simulation_engine.world_model.forward(
                        state, actions[:, t]
                    )
                    state = pred["next_state_mean"]
                    total_return += pred["reward_mean"].item()
                
                returns.append(total_return)
            
            returns = torch.tensor(returns).to(device)
            
            # Select elite sequences
            _, elite_indices = torch.topk(returns, self.num_elite)
            elite_sequences = action_sequences[elite_indices]
            
            # Update distribution
            mean = elite_sequences.mean(dim=0)
            std = elite_sequences.std(dim=0) + 1e-6
        
        return mean
