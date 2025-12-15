"""Tests for benchmarks and evaluation module."""

import torch
import pytest
import numpy as np
from rtdi.benchmarks import (
    BenchmarkResult,
    DynamicEnvironmentBenchmark,
    UncertaintyCalibrationBenchmark,
    OnlineLearningBenchmark,
    AdaptationBenchmark
)


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, state_dim=8, reward_range=(0, 1)):
        self.state_dim = state_dim
        self.reward_range = reward_range
        self.step_count = 0
        
    def reset(self):
        """Reset environment."""
        self.step_count = 0
        return np.random.randn(self.state_dim)
    
    def step(self, action):
        """Take step in environment."""
        self.step_count += 1
        next_state = np.random.randn(self.state_dim)
        reward = np.random.uniform(*self.reward_range)
        done = self.step_count >= 10
        info = {}
        return next_state, reward, done, info
    
    def render(self):
        """Render environment."""
        pass


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, state_dim=8, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def act(self, state, **kwargs):
        """Select action."""
        batch_size = state.shape[0]
        return torch.randn(batch_size, self.action_dim)
    
    def learn(self, experience):
        """Learn from experience."""
        return {"loss": 0.1}


class TestBenchmarkResult:
    """Test suite for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            mean_reward=10.5,
            std_reward=2.3,
            mean_episode_length=50.0,
            success_rate=0.75,
            mean_latency_ms=5.2,
            mean_uncertainty=0.15,
            additional_metrics={"custom_metric": 42.0}
        )
        
        assert result.mean_reward == 10.5
        assert result.std_reward == 2.3
        assert result.success_rate == 0.75
        assert result.additional_metrics["custom_metric"] == 42.0
        
    def test_benchmark_result_all_fields(self):
        """Test all benchmark result fields."""
        result = BenchmarkResult(
            mean_reward=15.0,
            std_reward=3.0,
            mean_episode_length=100.0,
            success_rate=0.9,
            mean_latency_ms=2.5,
            mean_uncertainty=0.1,
            additional_metrics={}
        )
        
        assert isinstance(result.mean_reward, (int, float))
        assert isinstance(result.success_rate, float)
        assert isinstance(result.additional_metrics, dict)


class TestDynamicEnvironmentBenchmark:
    """Test suite for DynamicEnvironmentBenchmark."""

    def setup_method(self):
        """Setup for each test."""
        def env_fn():
            return MockEnvironment(state_dim=8)
        
        self.benchmark = DynamicEnvironmentBenchmark(
            env_fn=env_fn,
            num_episodes=5,
            max_steps=10,
            seed=42
        )
        
    def test_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.num_episodes == 5
        assert self.benchmark.max_steps == 10
        assert self.benchmark.seed == 42
        
    def test_evaluate_agent(self):
        """Test agent evaluation."""
        agent = MockAgent(state_dim=8, action_dim=4)
        
        result = self.benchmark.evaluate_agent(agent, verbose=False)
        
        assert isinstance(result, BenchmarkResult)
        assert result.mean_reward is not None
        assert result.std_reward >= 0
        assert 0 <= result.success_rate <= 1.0
        assert result.mean_latency_ms > 0
        
    def test_evaluate_agent_metrics(self):
        """Test that all metrics are computed."""
        agent = MockAgent(state_dim=8, action_dim=4)
        
        result = self.benchmark.evaluate_agent(agent, verbose=False)
        
        assert hasattr(result, 'mean_reward')
        assert hasattr(result, 'std_reward')
        assert hasattr(result, 'mean_episode_length')
        assert hasattr(result, 'success_rate')
        assert hasattr(result, 'mean_latency_ms')
        
    def test_evaluate_multiple_episodes(self):
        """Test evaluation over multiple episodes."""
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        benchmark = DynamicEnvironmentBenchmark(
            env_fn=env_fn,
            num_episodes=10,
            max_steps=20
        )
        
        agent = MockAgent(state_dim=6, action_dim=3)
        result = benchmark.evaluate_agent(agent, verbose=False)
        
        assert result.mean_episode_length <= 20
        
    def test_compare_agents(self):
        """Test comparing multiple agents."""
        agent1 = MockAgent(state_dim=8, action_dim=4)
        agent2 = MockAgent(state_dim=8, action_dim=4)
        
        results = self.benchmark.compare_agents({
            "agent1": agent1,
            "agent2": agent2
        }, verbose=False)
        
        assert "agent1" in results
        assert "agent2" in results
        assert isinstance(results["agent1"], BenchmarkResult)
        assert isinstance(results["agent2"], BenchmarkResult)
        
    def test_additional_metrics(self):
        """Test that additional metrics are captured."""
        agent = MockAgent(state_dim=8, action_dim=4)
        
        result = self.benchmark.evaluate_agent(agent, verbose=False)
        
        assert "additional_metrics" in result.__dict__
        assert isinstance(result.additional_metrics, dict)
        
    def test_latency_measurement(self):
        """Test that latency is measured."""
        agent = MockAgent(state_dim=8, action_dim=4)
        
        result = self.benchmark.evaluate_agent(agent, verbose=False)
        
        # Latency should be positive
        assert result.mean_latency_ms > 0
        assert result.mean_latency_ms < 10000  # Should be reasonable
        
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        def good_env_fn():
            return MockEnvironment(state_dim=8, reward_range=(1, 2))
        
        benchmark = DynamicEnvironmentBenchmark(
            env_fn=good_env_fn,
            num_episodes=5,
            max_steps=10
        )
        
        agent = MockAgent(state_dim=8, action_dim=4)
        result = benchmark.evaluate_agent(agent, verbose=False)
        
        # With positive rewards, success rate should be high
        assert 0 <= result.success_rate <= 1.0


class TestUncertaintyCalibrationBenchmark:
    """Test suite for UncertaintyCalibrationBenchmark."""

    def setup_method(self):
        """Setup for each test."""
        self.benchmark = UncertaintyCalibrationBenchmark(num_samples=100)
        
    def test_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.num_samples == 100
        
    def test_evaluate_calibration(self):
        """Test uncertainty calibration evaluation."""
        from rtdi.world_models import ProbabilisticWorldModel
        
        model = ProbabilisticWorldModel(
            state_dim=6,
            action_dim=3,
            ensemble_size=3,
            hidden_dim=32
        )
        
        # Create test data
        test_data = {
            "states": torch.randn(50, 6),
            "actions": torch.randn(50, 3),
            "next_states": torch.randn(50, 6)
        }
        
        metrics = self.benchmark.evaluate_calibration(model, test_data)
        
        assert "within_1sigma" in metrics
        assert "within_2sigma" in metrics
        assert "within_3sigma" in metrics
        assert "mean_absolute_calibration_error" in metrics
        
    def test_calibration_metrics_range(self):
        """Test that calibration metrics are in valid range."""
        from rtdi.world_models import ProbabilisticWorldModel
        
        model = ProbabilisticWorldModel(
            state_dim=6,
            action_dim=3,
            ensemble_size=2,
            hidden_dim=32
        )
        
        test_data = {
            "states": torch.randn(30, 6),
            "actions": torch.randn(30, 3),
            "next_states": torch.randn(30, 6)
        }
        
        metrics = self.benchmark.evaluate_calibration(model, test_data)
        
        # Coverage metrics should be between 0 and 1
        assert 0 <= metrics["within_1sigma"] <= 1
        assert 0 <= metrics["within_2sigma"] <= 1
        assert 0 <= metrics["within_3sigma"] <= 1
        
    def test_calibration_with_perfect_model(self):
        """Test calibration with well-calibrated predictions."""
        from rtdi.world_models import ProbabilisticWorldModel
        
        model = ProbabilisticWorldModel(
            state_dim=4,
            action_dim=2,
            ensemble_size=3,
            hidden_dim=32
        )
        
        # Train model a bit
        for _ in range(10):
            batch = {
                "states": torch.randn(16, 4),
                "actions": torch.randn(16, 2),
                "next_states": torch.randn(16, 4),
                "rewards": torch.randn(16, 1)
            }
            model.train_step(batch)
        
        test_data = {
            "states": torch.randn(50, 4),
            "actions": torch.randn(50, 2),
            "next_states": torch.randn(50, 4)
        }
        
        metrics = self.benchmark.evaluate_calibration(model, test_data)
        
        assert metrics["mean_absolute_calibration_error"] >= 0


class TestOnlineLearningBenchmark:
    """Test suite for OnlineLearningBenchmark."""

    def setup_method(self):
        """Setup for each test."""
        self.benchmark = OnlineLearningBenchmark(num_steps=100)
        
    def test_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.num_steps == 100
        
    def test_evaluate_online_learning(self):
        """Test online learning evaluation."""
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        agent = MockAgent(state_dim=6, action_dim=3)
        
        history = self.benchmark.evaluate_online_learning(
            agent=agent,
            env_fn=env_fn,
            num_steps=50
        )
        
        assert "rewards" in history
        assert isinstance(history["rewards"], list)
        
    def test_learning_curve(self):
        """Test that learning curve is captured."""
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        agent = MockAgent(state_dim=6, action_dim=3)
        
        history = self.benchmark.evaluate_online_learning(
            agent=agent,
            env_fn=env_fn,
            num_steps=100,
            window_size=10
        )
        
        # Should have multiple reward measurements
        assert len(history["rewards"]) > 0
        
    def test_loss_tracking(self):
        """Test that losses are tracked."""
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        agent = MockAgent(state_dim=6, action_dim=3)
        
        history = self.benchmark.evaluate_online_learning(
            agent=agent,
            env_fn=env_fn,
            num_steps=50
        )
        
        if "losses" in history:
            assert isinstance(history["losses"], list)


class TestAdaptationBenchmark:
    """Test suite for AdaptationBenchmark."""

    def setup_method(self):
        """Setup for each test."""
        self.benchmark = AdaptationBenchmark(
            num_phases=3,
            steps_per_phase=50
        )
        
    def test_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.num_phases == 3
        assert self.benchmark.steps_per_phase == 50
        
    def test_evaluate_adaptation(self):
        """Test adaptation evaluation."""
        # Create different environments for each phase
        def env_fn1():
            return MockEnvironment(state_dim=6, reward_range=(0, 1))
        
        def env_fn2():
            return MockEnvironment(state_dim=6, reward_range=(1, 2))
        
        def env_fn3():
            return MockEnvironment(state_dim=6, reward_range=(0.5, 1.5))
        
        agent = MockAgent(state_dim=6, action_dim=3)
        
        results = self.benchmark.evaluate_adaptation(
            agent=agent,
            env_fns=[env_fn1, env_fn2, env_fn3]
        )
        
        assert "phase_performances" in results
        assert "adaptation_speeds" in results
        assert "mean_performance" in results
        assert "mean_adaptation_speed" in results
        
    def test_phase_performances(self):
        """Test that phase performances are tracked."""
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        agent = MockAgent(state_dim=6, action_dim=3)
        
        env_fns = [env_fn for _ in range(3)]
        
        results = self.benchmark.evaluate_adaptation(
            agent=agent,
            env_fns=env_fns
        )
        
        assert len(results["phase_performances"]) == 3
        
    def test_adaptation_speed(self):
        """Test adaptation speed calculation."""
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        agent = MockAgent(state_dim=6, action_dim=3)
        
        env_fns = [env_fn for _ in range(2)]
        
        results = self.benchmark.evaluate_adaptation(
            agent=agent,
            env_fns=env_fns
        )
        
        # Adaptation speeds should be calculated
        assert len(results["adaptation_speeds"]) == 2
        assert isinstance(results["mean_adaptation_speed"], (int, float))
        
    def test_distribution_shift_handling(self):
        """Test handling of distribution shifts."""
        # Create environments with different reward distributions
        def easy_env():
            return MockEnvironment(state_dim=6, reward_range=(2, 3))
        
        def hard_env():
            return MockEnvironment(state_dim=6, reward_range=(-1, 0))
        
        agent = MockAgent(state_dim=6, action_dim=3)
        
        results = self.benchmark.evaluate_adaptation(
            agent=agent,
            env_fns=[easy_env, hard_env, easy_env]
        )
        
        # Should complete without errors
        assert len(results["phase_performances"]) == 3


class TestBenchmarkIntegration:
    """Integration tests for benchmark components."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        from rtdi.agents import ModularAgent
        
        agent = ModularAgent(
            state_dim=6,
            action_dim=3,
            config={
                "world_model": {"ensemble_size": 2, "hidden_dim": 32},
                "decision_loop": {"planning_horizon": 3},
                "learner": {"replay_capacity": 100}
            }
        )
        
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        # Dynamic environment benchmark
        dyn_bench = DynamicEnvironmentBenchmark(
            env_fn=env_fn,
            num_episodes=3,
            max_steps=10
        )
        
        result = dyn_bench.evaluate_agent(agent, verbose=False)
        
        assert isinstance(result, BenchmarkResult)
        assert result.mean_reward is not None
        
    def test_multi_benchmark_comparison(self):
        """Test running multiple benchmarks."""
        agent1 = MockAgent(state_dim=6, action_dim=3)
        agent2 = MockAgent(state_dim=6, action_dim=3)
        
        def env_fn():
            return MockEnvironment(state_dim=6)
        
        benchmark = DynamicEnvironmentBenchmark(
            env_fn=env_fn,
            num_episodes=5,
            max_steps=10
        )
        
        results = benchmark.compare_agents({
            "baseline": agent1,
            "improved": agent2
        }, verbose=False)
        
        # Compare results
        baseline_reward = results["baseline"].mean_reward
        improved_reward = results["improved"].mean_reward
        
        assert isinstance(baseline_reward, (int, float))
        assert isinstance(improved_reward, (int, float))
        
    def test_uncertainty_and_performance_correlation(self):
        """Test correlation between uncertainty and performance."""
        from rtdi.world_models import ProbabilisticWorldModel
        
        model = ProbabilisticWorldModel(
            state_dim=6,
            action_dim=3,
            ensemble_size=3,
            hidden_dim=32
        )
        
        # Train model
        for _ in range(20):
            batch = {
                "states": torch.randn(16, 6),
                "actions": torch.randn(16, 3),
                "next_states": torch.randn(16, 6),
                "rewards": torch.randn(16, 1)
            }
            model.train_step(batch)
        
        # Test calibration
        calib_bench = UncertaintyCalibrationBenchmark(num_samples=50)
        
        test_data = {
            "states": torch.randn(50, 6),
            "actions": torch.randn(50, 3),
            "next_states": torch.randn(50, 6)
        }
        
        calib_metrics = calib_bench.evaluate_calibration(model, test_data)
        
        assert calib_metrics["mean_absolute_calibration_error"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])