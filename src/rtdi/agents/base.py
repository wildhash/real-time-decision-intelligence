"""Base agent architecture for modular design."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class Agent(ABC):
    """Abstract base class for intelligent agents."""

    @abstractmethod
    def act(self, observation: torch.Tensor, **kwargs) -> torch.Tensor:
        """Select action given observation.
        
        Args:
            observation: Current observation
            **kwargs: Additional arguments
            
        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Learn from experience.
        
        Args:
            experience: Experience dictionary
            
        Returns:
            Learning metrics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save agent to disk.
        
        Args:
            path: Save path
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Load agent from disk.
        
        Args:
            path: Load path
        """
        pass


class ModularAgent(Agent):
    """Modular agent combining world model, decision loops, and learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize modular agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: Configuration dictionary
        """
        from rtdi.world_models import ProbabilisticWorldModel
        from rtdi.decision_loops import HierarchicalDecisionLoop
        from rtdi.learning import HybridLearner
        from rtdi.uncertainty import UncertaintyEstimator
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Initialize world model
        self.world_model = ProbabilisticWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            **self.config.get("world_model", {})
        )
        
        # Initialize decision loops
        decision_config = self.config.get("decision_loop", {})
        self.decision_loop = HierarchicalDecisionLoop(
            state_dim=state_dim,
            action_dim=action_dim,
            world_model=self.world_model,
            reflex_config=decision_config.get("reflex_config"),
            deliberative_config=decision_config.get("deliberative_config"),
            meta_config=decision_config.get("meta_config")
        )
        
        # Initialize learning system
        self.learner = HybridLearner(
            model=self.world_model,
            **self.config.get("learner", {})
        )
        
        # Initialize uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            **self.config.get("uncertainty", {})
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_count = 0

    def act(
        self,
        observation: torch.Tensor,
        time_budget: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """Select action using hierarchical decision loop.
        
        Args:
            observation: Current observation
            time_budget: Time available for decision
            **kwargs: Additional context
            
        Returns:
            Selected action
        """
        observation = observation.to(self.device)
        
        # Estimate uncertainty in current state
        # For now, use a simple heuristic
        uncertainty = None
        
        # Make decision
        action, level = self.decision_loop.decide(
            observation,
            time_budget=time_budget,
            uncertainty=uncertainty,
            context=kwargs
        )
        
        self.step_count += 1
        
        return action

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Learn from experience using hybrid learning.
        
        Args:
            experience: Dictionary with state, action, reward, next_state, done
            
        Returns:
            Learning metrics
        """
        # Online learning
        metrics = self.learner.observe_online(
            state=experience["state"],
            action=experience["action"],
            reward=experience["reward"],
            next_state=experience["next_state"],
            done=experience["done"]
        )
        
        # Update decision loops (only if we have batch data with 'states' key)
        if "states" in experience:
            loop_metrics = self.decision_loop.update_all(experience)
            metrics.update(loop_metrics)
        
        return metrics

    def train_offline(self, num_epochs: int = 10) -> Dict[str, Any]:
        """Perform offline training on collected data.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Training history
        """
        return self.learner.train_offline(num_epochs=num_epochs)

    def save(self, path: str):
        """Save agent to disk.
        
        Args:
            path: Save path
        """
        checkpoint = {
            "world_model": self.world_model.ensemble.state_dict(),
            "reflex": self.decision_loop.reflex.policy.state_dict(),
            "meta": {
                "encoder": self.decision_loop.meta.context_encoder.state_dict(),
                "policy": self.decision_loop.meta.meta_policy.state_dict(),
            },
            "config": self.config,
            "step_count": self.step_count,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load agent from disk.
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model.ensemble.load_state_dict(checkpoint["world_model"])
        self.decision_loop.reflex.policy.load_state_dict(checkpoint["reflex"])
        self.decision_loop.meta.context_encoder.load_state_dict(checkpoint["meta"]["encoder"])
        self.decision_loop.meta.meta_policy.load_state_dict(checkpoint["meta"]["policy"])
        self.step_count = checkpoint["step_count"]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "step_count": self.step_count,
            "learner_stats": self.learner.get_stats(),
        }
