"""Hierarchical Decision Loops: Reflex, Deliberative, and Meta-Learning."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from enum import Enum
import time


class DecisionLevel(Enum):
    """Decision loop hierarchy levels."""
    REFLEX = 1  # Fast reactive decisions (<10ms)
    DELIBERATIVE = 2  # Planning-based decisions (<100ms)
    META = 3  # Learning and adaptation (>1s)


class DecisionLoop(ABC):
    """Abstract base class for decision loops."""

    @abstractmethod
    def decide(self, state: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Make a decision given current state.
        
        Args:
            state: Current state
            context: Additional context information
            
        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update the decision loop from experience.
        
        Args:
            experience: Experience dictionary
            
        Returns:
            Update metrics
        """
        pass


class ReflexLoop(DecisionLoop):
    """Fast reflex loop for immediate reactions (<10ms latency).
    
    Uses a simple lookup table or small neural network for low-latency decisions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3
    ):
        """Initialize reflex loop.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Small, fast network for reflex actions
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def decide(self, state: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Fast reflex decision.
        
        Args:
            state: Current state
            context: Context (ignored for reflex)
            
        Returns:
            Action tensor
        """
        state = state.to(self.device)
        with torch.no_grad():
            action = self.policy(state)
        return action

    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update reflex policy using experience.
        
        Args:
            experience: Dict with 'states', 'actions', 'advantages'
            
        Returns:
            Update metrics
        """
        states = experience["states"].to(self.device)
        actions = experience["actions"].to(self.device)
        advantages = experience["advantages"].to(self.device)
        
        # Compute policy loss (behavior cloning with advantage weighting)
        predicted_actions = self.policy(states)
        loss = ((predicted_actions - actions) ** 2 * advantages.unsqueeze(-1)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"reflex_loss": loss.item()}


class DeliberativeLoop(DecisionLoop):
    """Deliberative planning loop using model-based rollouts (<100ms latency).
    
    Plans ahead using world model predictions and evaluates action sequences.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        world_model: Any,
        planning_horizon: int = 5,
        num_candidates: int = 10,
        temperature: float = 1.0
    ):
        """Initialize deliberative loop.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            world_model: World model for planning
            planning_horizon: Number of steps to plan ahead
            num_candidates: Number of action sequences to evaluate
            temperature: Sampling temperature
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.world_model = world_model
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decide(self, state: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Plan-based decision using model predictive control.
        
        Args:
            state: Current state
            context: May contain goal, constraints, etc.
            
        Returns:
            First action of best plan
        """
        state = state.to(self.device)
        batch_size = state.shape[0]
        
        # Sample candidate action sequences
        action_sequences = torch.randn(
            self.num_candidates, batch_size, self.planning_horizon, self.action_dim
        ).to(self.device) * self.temperature
        action_sequences = torch.tanh(action_sequences)  # Bound actions
        
        # Evaluate each candidate using world model rollouts
        values = []
        for i in range(self.num_candidates):
            actions = action_sequences[i]  # [batch, horizon, action_dim]
            
            # Rollout using world model
            trajectory = self.world_model.sample_trajectories(
                state, actions, num_samples=1
            )
            
            # Compute value as sum of predicted rewards
            rewards = trajectory["rewards"].squeeze(0)  # [batch, horizon, 1]
            value = rewards.sum(dim=1)  # [batch, 1]
            values.append(value)
        
        values = torch.stack(values, dim=0)  # [num_candidates, batch, 1]
        
        # Select best action sequence
        best_idx = values.argmax(dim=0).squeeze(-1)  # [batch]
        
        # Extract first action from best sequences
        best_actions = torch.stack([
            action_sequences[best_idx[b], b, 0]
            for b in range(batch_size)
        ])
        
        return best_actions

    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update planning parameters (currently no-op, world model updated separately).
        
        Args:
            experience: Experience dictionary
            
        Returns:
            Empty metrics dict
        """
        # Planning uses world model which is updated separately
        return {}


class MetaLearningLoop(DecisionLoop):
    """Meta-learning loop for slow adaptation and strategy learning.
    
    Learns task distributions, updates learning rates, and adapts strategies.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int = 32,
        learning_rate: float = 1e-4
    ):
        """Initialize meta-learning loop.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            context_dim: Task context embedding dimension
            learning_rate: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        
        # Context encoder for task identification
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 128),  # state, action, reward
            nn.ReLU(),
            nn.Linear(128, context_dim)
        )
        
        # Meta-policy conditioned on task context
        self.meta_policy = nn.Sequential(
            nn.Linear(state_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.context_encoder.parameters()) + list(self.meta_policy.parameters()),
            lr=learning_rate
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_encoder.to(self.device)
        self.meta_policy.to(self.device)
        
        # Task context buffer
        self.task_buffer = []

    def encode_task_context(self, trajectories: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode task from trajectory history.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Task context embedding
        """
        # Concatenate recent experiences
        experiences = []
        for traj in trajectories[-10:]:  # Use last 10 trajectories
            state = traj["state"]
            action = traj["action"]
            reward = traj["reward"]
            exp = torch.cat([state, action, reward.unsqueeze(-1)], dim=-1)
            experiences.append(exp)
        
        if len(experiences) == 0:
            # Return zero context if no history
            return torch.zeros(1, self.context_dim).to(self.device)
        
        experiences = torch.cat(experiences, dim=0)
        context = self.context_encoder(experiences).mean(dim=0, keepdim=True)
        return context

    def decide(self, state: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Meta-level decision with task adaptation.
        
        Args:
            state: Current state
            context: Must contain 'trajectories' for task inference
            
        Returns:
            Action tensor
        """
        state = state.to(self.device)
        
        # Infer task context
        trajectories = context.get("trajectories", [])
        task_context = self.encode_task_context(trajectories)
        
        # Replicate context for batch
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        task_context = task_context.repeat(state.shape[0], 1)
        
        # Meta-policy decision
        state_context = torch.cat([state, task_context], dim=-1)
        action = self.meta_policy(state_context)
        
        return action

    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Meta-learning update.
        
        Args:
            experience: Dict with task trajectories
            
        Returns:
            Update metrics
        """
        # Meta-learning: learn to adapt quickly to new tasks
        # This is a simplified version - full MAML would require inner/outer loops
        
        if "task_trajectories" not in experience:
            return {}
        
        task_trajectories = experience["task_trajectories"]
        
        total_loss = 0.0
        for task_data in task_trajectories:
            states = task_data["states"].to(self.device)
            actions = task_data["actions"].to(self.device)
            
            # Encode task context from support set
            task_context = self.encode_task_context([task_data])
            task_context = task_context.repeat(states.shape[0], 1)
            
            # Predict actions
            state_context = torch.cat([states, task_context], dim=-1)
            predicted_actions = self.meta_policy(state_context)
            
            # Imitation loss
            loss = ((predicted_actions - actions) ** 2).mean()
            total_loss += loss
        
        if len(task_trajectories) > 0:
            total_loss = total_loss / len(task_trajectories)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return {"meta_loss": total_loss.item()}
        
        return {}


class HierarchicalDecisionLoop:
    """Hierarchical decision system with reflex, deliberative, and meta loops.
    
    Routes decisions through appropriate loops based on time constraints,
    uncertainty, and task complexity.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        world_model: Any,
        reflex_config: Optional[Dict[str, Any]] = None,
        deliberative_config: Optional[Dict[str, Any]] = None,
        meta_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize hierarchical decision loop.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            world_model: World model for deliberative planning
            reflex_config: Configuration for reflex loop
            deliberative_config: Configuration for deliberative loop
            meta_config: Configuration for meta-learning loop
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize decision loops
        self.reflex = ReflexLoop(state_dim, action_dim, **(reflex_config or {}))
        self.deliberative = DeliberativeLoop(
            state_dim, action_dim, world_model, **(deliberative_config or {})
        )
        self.meta = MetaLearningLoop(state_dim, action_dim, **(meta_config or {}))
        
        # Decision routing policy
        self.time_budgets = {
            DecisionLevel.REFLEX: 0.01,  # 10ms
            DecisionLevel.DELIBERATIVE: 0.1,  # 100ms
            DecisionLevel.META: 1.0,  # 1s
        }

    def decide(
        self,
        state: torch.Tensor,
        time_budget: float = 0.1,
        uncertainty: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, DecisionLevel]:
        """Make hierarchical decision.
        
        Args:
            state: Current state
            time_budget: Available time for decision (seconds)
            uncertainty: State uncertainty estimate
            context: Additional context
            
        Returns:
            Tuple of (action, decision_level_used)
        """
        context = context or {}
        start_time = time.time()
        
        # Route to appropriate level based on time budget and uncertainty
        if time_budget < self.time_budgets[DecisionLevel.REFLEX]:
            # Use fast reflex
            action = self.reflex.decide(state, context)
            level = DecisionLevel.REFLEX
        elif time_budget < self.time_budgets[DecisionLevel.DELIBERATIVE]:
            # Use deliberative planning if enough time
            action = self.deliberative.decide(state, context)
            level = DecisionLevel.DELIBERATIVE
        else:
            # Use meta-learning for long-term adaptation
            action = self.meta.decide(state, context)
            level = DecisionLevel.META
        
        elapsed = time.time() - start_time
        
        # Fallback to faster loop if time exceeded
        if elapsed > time_budget and level != DecisionLevel.REFLEX:
            action = self.reflex.decide(state, context)
            level = DecisionLevel.REFLEX
        
        return action, level

    def update_all(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update all decision loops.
        
        Args:
            experience: Experience dictionary
            
        Returns:
            Combined metrics
        """
        metrics = {}
        
        # Update each loop
        metrics.update(self.reflex.update(experience))
        metrics.update(self.deliberative.update(experience))
        metrics.update(self.meta.update(experience))
        
        return metrics
