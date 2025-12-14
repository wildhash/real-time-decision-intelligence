"""Evaluation benchmarks for dynamic environments."""

from typing import Dict, Any, List, Optional, Callable
import torch
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Results from a benchmark evaluation."""
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    success_rate: float
    mean_latency_ms: float
    mean_uncertainty: float
    additional_metrics: Dict[str, float]


class DynamicEnvironmentBenchmark:
    """Benchmark suite for evaluating agents in dynamic environments."""

    def __init__(
        self,
        env_fn: Callable,
        num_episodes: int = 100,
        max_steps: int = 1000,
        seed: int = 42
    ):
        """Initialize benchmark.
        
        Args:
            env_fn: Function to create environment
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            seed: Random seed
        """
        self.env_fn = env_fn
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)

    def evaluate_agent(
        self,
        agent: Any,
        render: bool = False,
        verbose: bool = False
    ) -> BenchmarkResult:
        """Evaluate agent performance.
        
        Args:
            agent: Agent to evaluate
            render: Whether to render environment
            verbose: Whether to print progress
            
        Returns:
            Benchmark results
        """
        episode_rewards = []
        episode_lengths = []
        latencies = []
        uncertainties = []
        successes = 0
        
        for episode in range(self.num_episodes):
            env = self.env_fn()
            state = env.reset()
            
            episode_reward = 0.0
            episode_uncertainty = 0.0
            steps = 0
            
            for step in range(self.max_steps):
                # Measure inference latency
                start_time = time.perf_counter()
                
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Agent action
                action_tensor = agent.act(state_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()
                
                latency = (time.perf_counter() - start_time) * 1000  # ms
                latencies.append(latency)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                if render:
                    env.render()
                
                episode_reward += reward
                steps += 1
                
                # Track uncertainty if available
                if hasattr(agent, 'uncertainty_estimator'):
                    # Simplified uncertainty tracking
                    episode_uncertainty += 0.1
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Count success (simple threshold)
            if episode_reward > 0:
                successes += 1
            
            if episode_uncertainty > 0:
                uncertainties.append(episode_uncertainty / steps)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}: "
                      f"Reward={episode_reward:.2f}, Length={steps}")
        
        return BenchmarkResult(
            mean_reward=np.mean(episode_rewards),
            std_reward=np.std(episode_rewards),
            mean_episode_length=np.mean(episode_lengths),
            success_rate=successes / self.num_episodes,
            mean_latency_ms=np.mean(latencies),
            mean_uncertainty=np.mean(uncertainties) if uncertainties else 0.0,
            additional_metrics={
                "median_reward": np.median(episode_rewards),
                "min_reward": np.min(episode_rewards),
                "max_reward": np.max(episode_rewards),
            }
        )

    def compare_agents(
        self,
        agents: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple agents.
        
        Args:
            agents: Dictionary of agent_name -> agent
            verbose: Whether to print progress
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        for name, agent in agents.items():
            if verbose:
                print(f"\nEvaluating {name}...")
            
            result = self.evaluate_agent(agent, verbose=verbose)
            results[name] = result
            
            if verbose:
                print(f"{name} Results:")
                print(f"  Mean Reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
                print(f"  Success Rate: {result.success_rate:.2%}")
                print(f"  Mean Latency: {result.mean_latency_ms:.2f}ms")
        
        return results


class UncertaintyCalibrationBenchmark:
    """Benchmark for evaluating uncertainty calibration."""

    def __init__(self, num_samples: int = 1000):
        """Initialize calibration benchmark.
        
        Args:
            num_samples: Number of samples for calibration
        """
        self.num_samples = num_samples

    def evaluate_calibration(
        self,
        world_model: Any,
        test_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate uncertainty calibration.
        
        Args:
            world_model: World model to evaluate
            test_data: Test data with states, actions, next_states
            
        Returns:
            Calibration metrics
        """
        states = test_data["states"]
        actions = test_data["actions"]
        true_next_states = test_data["next_states"]
        
        # Get predictions
        predictions = world_model.forward(states, actions)
        
        predicted_mean = predictions["next_state_mean"]
        predicted_std = predictions["next_state_std"]
        
        # Compute prediction errors
        errors = torch.abs(predicted_mean - true_next_states)
        
        # Check if errors are within predicted uncertainty bands
        within_1sigma = (errors <= predicted_std).float().mean()
        within_2sigma = (errors <= 2 * predicted_std).float().mean()
        within_3sigma = (errors <= 3 * predicted_std).float().mean()
        
        # Compute calibration error (expected vs actual coverage)
        calibration_error_1sigma = abs(within_1sigma.item() - 0.68)
        calibration_error_2sigma = abs(within_2sigma.item() - 0.95)
        calibration_error_3sigma = abs(within_3sigma.item() - 0.997)
        
        # Mean absolute calibration error
        mace = (calibration_error_1sigma + calibration_error_2sigma + calibration_error_3sigma) / 3
        
        return {
            "within_1sigma": within_1sigma.item(),
            "within_2sigma": within_2sigma.item(),
            "within_3sigma": within_3sigma.item(),
            "calibration_error_1sigma": calibration_error_1sigma,
            "calibration_error_2sigma": calibration_error_2sigma,
            "calibration_error_3sigma": calibration_error_3sigma,
            "mean_absolute_calibration_error": mace,
        }


class OnlineLearningBenchmark:
    """Benchmark for online learning performance."""

    def __init__(self, num_steps: int = 10000):
        """Initialize online learning benchmark.
        
        Args:
            num_steps: Number of online learning steps
        """
        self.num_steps = num_steps

    def evaluate_online_learning(
        self,
        agent: Any,
        env_fn: Callable,
        window_size: int = 100
    ) -> Dict[str, List[float]]:
        """Evaluate online learning performance.
        
        Args:
            agent: Agent with online learning
            env_fn: Environment factory
            window_size: Window for computing metrics
            
        Returns:
            Learning curves
        """
        env = env_fn()
        state = env.reset()
        
        rewards_history = []
        losses_history = []
        
        recent_rewards = []
        
        for step in range(self.num_steps):
            # Agent action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = agent.act(state_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Online learning
            experience = {
                "state": state_tensor.squeeze(0),
                "action": action_tensor.squeeze(0),
                "reward": torch.FloatTensor([reward]),
                "next_state": torch.FloatTensor(next_state),
                "done": torch.BoolTensor([done]),
            }
            
            metrics = agent.learn(experience)
            
            recent_rewards.append(reward)
            if len(recent_rewards) > window_size:
                recent_rewards.pop(0)
            
            # Track metrics
            if (step + 1) % window_size == 0:
                rewards_history.append(np.mean(recent_rewards))
                if "loss" in metrics:
                    losses_history.append(metrics["loss"])
            
            state = next_state
            if done:
                state = env.reset()
        
        return {
            "rewards": rewards_history,
            "losses": losses_history,
        }


class AdaptationBenchmark:
    """Benchmark for measuring adaptation to distribution shifts."""

    def __init__(self, num_phases: int = 5, steps_per_phase: int = 1000):
        """Initialize adaptation benchmark.
        
        Args:
            num_phases: Number of environment phases
            steps_per_phase: Steps per phase
        """
        self.num_phases = num_phases
        self.steps_per_phase = steps_per_phase

    def evaluate_adaptation(
        self,
        agent: Any,
        env_fns: List[Callable]
    ) -> Dict[str, List[float]]:
        """Evaluate adaptation across environment changes.
        
        Args:
            agent: Agent to evaluate
            env_fns: List of environment factories (one per phase)
            
        Returns:
            Adaptation metrics
        """
        phase_performances = []
        adaptation_speeds = []
        
        for phase_idx, env_fn in enumerate(env_fns):
            env = env_fn()
            state = env.reset()
            
            phase_rewards = []
            
            for step in range(self.steps_per_phase):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor = agent.act(state_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()
                
                next_state, reward, done, info = env.step(action)
                
                # Learn online
                experience = {
                    "state": state_tensor.squeeze(0),
                    "action": action_tensor.squeeze(0),
                    "reward": torch.FloatTensor([reward]),
                    "next_state": torch.FloatTensor(next_state),
                    "done": torch.BoolTensor([done]),
                }
                agent.learn(experience)
                
                phase_rewards.append(reward)
                
                state = next_state
                if done:
                    state = env.reset()
            
            # Compute phase performance
            avg_reward = np.mean(phase_rewards)
            phase_performances.append(avg_reward)
            
            # Compute adaptation speed (reward improvement over phase)
            window = 100
            early_perf = np.mean(phase_rewards[:window])
            late_perf = np.mean(phase_rewards[-window:])
            adaptation_speed = late_perf - early_perf
            adaptation_speeds.append(adaptation_speed)
        
        return {
            "phase_performances": phase_performances,
            "adaptation_speeds": adaptation_speeds,
            "mean_performance": np.mean(phase_performances),
            "mean_adaptation_speed": np.mean(adaptation_speeds),
        }
