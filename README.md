# Real-Time Decision Intelligence

A production-ready machine learning framework for probabilistic world modeling, uncertainty-aware reasoning, and real-time decision-making in dynamic, high-uncertainty environments.

## Features

### Core Capabilities
- **Probabilistic World Models**: Ensemble-based models with epistemic and aleatoric uncertainty estimation
- **Hierarchical Decision Loops**: 
  - Fast reflex loop (<10ms latency)
  - Deliberative planning loop (<100ms latency)
  - Meta-learning loop for long-term adaptation
- **Hybrid Learning**: Combined online and offline learning with prioritized replay
- **Simulation Engine**: Model-based planning with rollout capabilities
- **Reflexion-Based Self-Improvement**: Learn from failures with self-reflection
- **Low-Latency Inference**: Optimized inference with JIT, batching, and caching
- **Comprehensive Benchmarks**: Evaluation suite for dynamic environments

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## Quick Start

### Basic Agent Usage

```python
import torch
from rtdi import ModularAgent

# Create agent
agent = ModularAgent(
    state_dim=10,
    action_dim=4,
    config={
        "world_model": {"ensemble_size": 5, "hidden_dim": 256},
        "decision_loop": {"planning_horizon": 10},
        "learner": {"replay_capacity": 100000}
    }
)

# Interact with environment
state = torch.randn(1, 10)  # Current state
action = agent.act(state, time_budget=0.05)  # Get action with 50ms budget

# Learn from experience
experience = {
    "state": state.squeeze(0),
    "action": action.squeeze(0),
    "reward": torch.tensor([1.0]),
    "next_state": torch.randn(10),
    "done": torch.tensor([False])
}
metrics = agent.learn(experience)

# Save/load agent
agent.save("agent_checkpoint.pt")
agent.load("agent_checkpoint.pt")
```

### World Model Training

```python
from rtdi.world_models import ProbabilisticWorldModel

# Create world model
world_model = ProbabilisticWorldModel(
    state_dim=10,
    action_dim=4,
    ensemble_size=5,
    hidden_dim=256
)

# Train on batch
batch = {
    "states": torch.randn(32, 10),
    "actions": torch.randn(32, 4),
    "next_states": torch.randn(32, 10),
    "rewards": torch.randn(32, 1)
}
metrics = world_model.train_step(batch)

# Make predictions with uncertainty
prediction = world_model.forward(
    state=torch.randn(1, 10),
    action=torch.randn(1, 4)
)
# Returns: next_state_mean, next_state_std, reward_mean, reward_std,
#          epistemic_uncertainty, aleatoric_uncertainty
```

### Hierarchical Decision Making

```python
from rtdi.decision_loops import HierarchicalDecisionLoop

decision_loop = HierarchicalDecisionLoop(
    state_dim=10,
    action_dim=4,
    world_model=world_model
)

# Fast reflex decision (< 10ms)
action, level = decision_loop.decide(
    state=torch.randn(1, 10),
    time_budget=0.01  # 10ms
)

# Deliberative planning (< 100ms)
action, level = decision_loop.decide(
    state=torch.randn(1, 10),
    time_budget=0.1  # 100ms
)

# Meta-learning decision
action, level = decision_loop.decide(
    state=torch.randn(1, 10),
    time_budget=1.0,  # 1s
    context={"trajectories": past_trajectories}
)
```

### Simulation and Planning

```python
from rtdi.simulation import SimulationEngine, RolloutConfig

# Create simulation engine
sim_engine = SimulationEngine(world_model=world_model)

# Rollout policy
def simple_policy(state):
    return torch.randn(state.shape[0], 4)

config = RolloutConfig(horizon=10, num_samples=5, discount=0.99)
trajectory = sim_engine.rollout(
    initial_state=torch.randn(1, 10),
    policy=simple_policy,
    config=config
)
# Returns: states, actions, rewards, returns, uncertainties
```

### Reflexion-Based Learning

```python
from rtdi.reflexion import ReflexionAgent, SelfImprovementLoop

# Create reflexion agent
reflex_agent = ReflexionAgent(
    state_dim=10,
    action_dim=4,
    memory_capacity=1000
)

# Self-improvement loop
improvement_loop = SelfImprovementLoop(
    agent=reflex_agent,
    improvement_frequency=10
)

# Run episode with reflection
result = improvement_loop.run_episode(env, max_steps=100)
print(result["reflection"])  # Get reflection on performance
```

### Uncertainty Estimation

```python
from rtdi.uncertainty import UncertaintyEstimator

estimator = UncertaintyEstimator(method="ensemble", num_samples=10)

# Ensemble predictions from multiple models
predictions = torch.stack([model(x) for model in models])

# Compute uncertainty
uncertainty = estimator.estimate(predictions)
# Returns: mean, std, variance, confidence, disagreement, epistemic_uncertainty

# Check confidence
is_confident = estimator.is_high_confidence(uncertainty["std"])
```

### Benchmarking

```python
from rtdi.benchmarks import DynamicEnvironmentBenchmark

# Create benchmark
benchmark = DynamicEnvironmentBenchmark(
    env_fn=lambda: gym.make("CartPole-v1"),
    num_episodes=100
)

# Evaluate agent
results = benchmark.evaluate_agent(agent, verbose=True)
print(f"Mean Reward: {results.mean_reward:.2f}")
print(f"Success Rate: {results.success_rate:.2%}")
print(f"Mean Latency: {results.mean_latency_ms:.2f}ms")

# Compare multiple agents
results = benchmark.compare_agents({
    "agent_v1": agent1,
    "agent_v2": agent2
})
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Modular Agent                        │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌────────────────────────┐  │
│  │ World Model      │◄────►│ Decision Loops         │  │
│  │ - Ensemble       │      │ - Reflex (<10ms)      │  │
│  │ - Uncertainty    │      │ - Deliberative (<100ms)│  │
│  └──────────────────┘      │ - Meta (>1s)          │  │
│           ▲                └────────────────────────┘  │
│           │                          ▲                  │
│  ┌────────┴─────────┐               │                  │
│  │ Learning System  │               │                  │
│  │ - Online         │               │                  │
│  │ - Offline        │      ┌────────┴──────────┐      │
│  │ - Hybrid         │      │ Reflexion         │      │
│  └──────────────────┘      │ - Memory          │      │
│                             │ - Self-Improvement│      │
│  ┌──────────────────┐      └───────────────────┘      │
│  │ Inference Engine │                                  │
│  │ - JIT Compile    │                                  │
│  │ - Batching       │                                  │
│  │ - Caching        │                                  │
│  └──────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

### Decision Loop Hierarchy

1. **Reflex Loop** (<10ms): Fast reactive decisions using small neural networks
2. **Deliberative Loop** (<100ms): Model-based planning with world model rollouts
3. **Meta-Learning Loop** (>1s): Task adaptation and strategy learning

### Uncertainty Types

- **Epistemic Uncertainty**: Model uncertainty from limited knowledge (ensemble variance)
- **Aleatoric Uncertainty**: Data uncertainty inherent in environment (predicted variance)

## Advanced Usage

### Custom World Models

```python
from rtdi.world_models import WorldModel

class CustomWorldModel(WorldModel):
    def forward(self, state, action):
        # Custom prediction logic
        pass
    
    def train_step(self, batch):
        # Custom training logic
        pass
```

### Low-Latency Optimization

```python
from rtdi.inference import LowLatencyInferenceEngine

# Optimize for low latency
inference_engine = LowLatencyInferenceEngine(
    model=world_model.ensemble[0],
    config={
        "use_jit": True,
        "use_half_precision": True,
        "use_cache": True,
        "cache_capacity": 1000
    }
)

# Fast inference
output = inference_engine.infer(input_data)

# Benchmark performance
results = inference_engine.benchmark(num_iterations=100)
print(f"P95 Latency: {results['p95_latency_ms']:.2f}ms")
```

### Offline Dataset Training

```python
from rtdi.learning import OfflineLearner, ReplayBuffer

# Create dataset
dataset = ReplayBuffer(capacity=100000)
# ... populate dataset ...

# Offline training
offline_learner = OfflineLearner(
    model=world_model,
    batch_size=256,
    num_epochs=100
)
history = offline_learner.train(dataset)
```

## Performance Characteristics

- **Reflex Loop Latency**: <10ms (CPU), <5ms (GPU)
- **Deliberative Loop Latency**: 50-100ms with 10-step horizon
- **World Model Training**: ~1000 samples/sec on GPU
- **Inference Throughput**: >10,000 inferences/sec with batching
- **Memory Efficiency**: ~100MB for standard configuration

## Citation

```bibtex
@software{real_time_decision_intelligence,
  title = {Real-Time Decision Intelligence Framework},
  author = {Real-Time Decision Intelligence Team},
  year = {2024},
  url = {https://github.com/wildhash/real-time-decision-intelligence}
}
```

## License

MIT License - see LICENSE file for details
