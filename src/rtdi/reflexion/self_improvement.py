"""Reflexion-based self-improvement system for learning from failures."""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import deque


@dataclass
class Experience:
    """Experience data structure."""
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: bool
    reflection: Optional[str] = None
    success: Optional[bool] = None


class ReflexionMemory:
    """Memory system for storing and retrieving past experiences with reflections."""

    def __init__(self, capacity: int = 1000):
        """Initialize reflexion memory.
        
        Args:
            capacity: Maximum memory capacity
        """
        self.capacity = capacity
        self.successes = deque(maxlen=capacity)
        self.failures = deque(maxlen=capacity)
        self.reflections = deque(maxlen=capacity)

    def add_experience(self, experience: Experience):
        """Add experience with success/failure label.
        
        Args:
            experience: Experience to add
        """
        if experience.success:
            self.successes.append(experience)
        else:
            self.failures.append(experience)
        
        if experience.reflection:
            self.reflections.append({
                "experience": experience,
                "reflection": experience.reflection
            })

    def get_similar_failures(
        self,
        current_state: torch.Tensor,
        k: int = 5
    ) -> List[Experience]:
        """Retrieve similar past failures for learning.
        
        Args:
            current_state: Current state
            k: Number of similar experiences to retrieve
            
        Returns:
            List of similar failure experiences
        """
        if len(self.failures) == 0:
            return []
        
        # Compute similarity based on state distance
        similarities = []
        for exp in self.failures:
            distance = torch.norm(exp.state - current_state)
            similarities.append((distance.item(), exp))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0])
        return [exp for _, exp in similarities[:k]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "num_successes": len(self.successes),
            "num_failures": len(self.failures),
            "num_reflections": len(self.reflections),
            "success_rate": len(self.successes) / (len(self.successes) + len(self.failures))
                           if (len(self.successes) + len(self.failures)) > 0 else 0.0
        }


class ReflexionAgent:
    """Agent with reflexion-based self-improvement capabilities.
    
    Implements:
    1. Trial and error with memory
    2. Reflection on failures
    3. Strategy adaptation based on reflections
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        memory_capacity: int = 1000,
        learning_rate: float = 1e-3
    ):
        """Initialize reflexion agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            memory_capacity: Memory capacity
            learning_rate: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Value network for self-evaluation
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Reflexion memory
        self.memory = ReflexionMemory(capacity=memory_capacity)
        
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=learning_rate
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value.to(self.device)

    def act(self, state: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """Select action with optional exploration.
        
        Args:
            state: Current state
            explore: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state = state.to(self.device)
        
        with torch.no_grad():
            action = self.policy(state)
            
            if explore:
                # Add exploration noise
                noise = torch.randn_like(action) * 0.1
                action = action + noise
                action = torch.clamp(action, -1, 1)
        
        return action

    def reflect(self, trajectory: List[Experience]) -> str:
        """Generate reflection on trajectory outcome.
        
        Args:
            trajectory: List of experiences from episode
            
        Returns:
            Reflection text
        """
        # Analyze trajectory
        total_reward = sum(exp.reward.item() for exp in trajectory)
        trajectory_length = len(trajectory)
        
        # Simple rule-based reflection (could be enhanced with LLM)
        if total_reward < 0:
            reflection = f"Failed with reward {total_reward:.2f}. "
            
            # Analyze state-action patterns
            high_uncertainty_steps = sum(
                1 for exp in trajectory
                if hasattr(exp, 'uncertainty') and exp.uncertainty > 0.5
            )
            
            if high_uncertainty_steps > trajectory_length * 0.5:
                reflection += "High uncertainty in many steps suggests insufficient exploration. "
            
            if trajectory_length < 10:
                reflection += "Episode terminated too early - likely took risky action. "
            else:
                reflection += "Long episode but poor reward - strategy needs refinement. "
        else:
            reflection = f"Success with reward {total_reward:.2f}. "
            reflection += "Strategy worked well - reinforce these behaviors. "
        
        return reflection

    def learn_from_reflection(
        self,
        trajectory: List[Experience],
        reflection: str
    ) -> Dict[str, float]:
        """Learn from trajectory and reflection.
        
        Args:
            trajectory: Trajectory of experiences
            reflection: Reflection on trajectory
            
        Returns:
            Learning metrics
        """
        # Extract data from trajectory
        states = torch.stack([exp.state for exp in trajectory]).to(self.device)
        actions = torch.stack([exp.action for exp in trajectory]).to(self.device)
        rewards = torch.stack([exp.reward for exp in trajectory]).to(self.device)
        
        # Compute advantages
        values = self.value(states).squeeze(-1)
        returns = self._compute_returns(rewards)
        advantages = returns - values.detach()
        
        # Policy loss (weighted by advantages)
        predicted_actions = self.policy(states)
        policy_loss = ((predicted_actions - actions) ** 2 * advantages.unsqueeze(-1).abs()).mean()
        
        # Value loss
        value_loss = ((values - returns) ** 2).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Store in memory
        success = rewards.sum().item() > 0
        for exp in trajectory:
            exp.success = success
            exp.reflection = reflection
            self.memory.add_experience(exp)
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": loss.item(),
            "mean_advantage": advantages.mean().item(),
        }

    def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns.
        
        Args:
            rewards: Reward tensor [T]
            gamma: Discount factor
            
        Returns:
            Returns tensor [T]
        """
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns

    def improve_from_failures(self, num_samples: int = 10) -> Dict[str, float]:
        """Improve policy by learning from past failures.
        
        Args:
            num_samples: Number of failure experiences to learn from
            
        Returns:
            Improvement metrics
        """
        if len(self.memory.failures) < num_samples:
            return {}
        
        # Sample failures
        failures = list(self.memory.failures)[-num_samples:]
        
        states = torch.stack([f.state for f in failures]).to(self.device)
        actions = torch.stack([f.action for f in failures]).to(self.device)
        
        # Generate better actions (inverse of failed actions with exploration)
        with torch.no_grad():
            current_actions = self.policy(states)
        
        # Encourage different actions from failures
        # Use a contrastive loss to push away from failed actions
        predicted_actions = self.policy(states)
        
        # Maximize distance from failed actions
        distance = torch.norm(predicted_actions - actions, dim=-1)
        loss = -distance.mean()  # Negative because we want to maximize distance
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "failure_learning_loss": loss.item(),
            "mean_action_distance": distance.mean().item(),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance.
        
        Returns:
            Performance summary
        """
        mem_stats = self.memory.get_statistics()
        
        return {
            "memory_stats": mem_stats,
            "policy_parameters": sum(p.numel() for p in self.policy.parameters()),
            "value_parameters": sum(p.numel() for p in self.value.parameters()),
        }


class SelfImprovementLoop:
    """Main loop for reflexion-based self-improvement."""

    def __init__(
        self,
        agent: ReflexionAgent,
        improvement_frequency: int = 10,
        min_failures_for_learning: int = 5
    ):
        """Initialize self-improvement loop.
        
        Args:
            agent: Reflexion agent
            improvement_frequency: Episodes between improvement updates
            min_failures_for_learning: Minimum failures needed for learning
        """
        self.agent = agent
        self.improvement_frequency = improvement_frequency
        self.min_failures_for_learning = min_failures_for_learning
        self.episode_count = 0

    def run_episode(
        self,
        env,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """Run one episode with reflexion.
        
        Args:
            env: Environment to interact with
            max_steps: Maximum episode steps
            
        Returns:
            Episode results
        """
        trajectory = []
        state = env.reset()
        total_reward = 0.0
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Select action
            action_tensor = self.agent.act(state_tensor, explore=True)
            action = action_tensor.squeeze(0).cpu().numpy()
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            exp = Experience(
                state=state_tensor.squeeze(0),
                action=action_tensor.squeeze(0),
                reward=torch.FloatTensor([reward]),
                next_state=torch.FloatTensor(next_state),
                done=done
            )
            trajectory.append(exp)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Generate reflection
        reflection = self.agent.reflect(trajectory)
        
        # Learn from experience
        metrics = self.agent.learn_from_reflection(trajectory, reflection)
        
        self.episode_count += 1
        
        # Periodic self-improvement from failures
        if self.episode_count % self.improvement_frequency == 0:
            if len(self.agent.memory.failures) >= self.min_failures_for_learning:
                improvement_metrics = self.agent.improve_from_failures()
                metrics.update(improvement_metrics)
        
        return {
            "total_reward": total_reward,
            "episode_length": len(trajectory),
            "reflection": reflection,
            "metrics": metrics,
        }
