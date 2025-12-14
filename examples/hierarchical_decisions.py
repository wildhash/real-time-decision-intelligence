"""Example: Demonstrating hierarchical decision loops."""

import torch
import time
from rtdi.world_models import ProbabilisticWorldModel
from rtdi.decision_loops import HierarchicalDecisionLoop, DecisionLevel


def main():
    """Demonstrate hierarchical decision loops."""
    print("=" * 60)
    print("Hierarchical Decision Loops Example")
    print("=" * 60)
    
    # Setup
    state_dim = 8
    action_dim = 3
    
    # Create world model
    print("\n1. Creating world model...")
    world_model = ProbabilisticWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        ensemble_size=3,
        hidden_dim=64
    )
    print("   World model created")
    
    # Create hierarchical decision loop
    print("\n2. Creating hierarchical decision loop...")
    decision_loop = HierarchicalDecisionLoop(
        state_dim=state_dim,
        action_dim=action_dim,
        world_model=world_model,
        reflex_config={"hidden_dim": 32},
        deliberative_config={"planning_horizon": 5, "num_candidates": 10},
        meta_config={"context_dim": 16}
    )
    print("   Decision loop created with 3 levels:")
    print("   - Reflex: <10ms")
    print("   - Deliberative: <100ms")
    print("   - Meta: >1s")
    
    # Test state
    state = torch.randn(1, state_dim)
    
    # Test different decision levels
    print("\n3. Testing decision levels...")
    
    # Reflex decision
    print("\n   a) Reflex Loop (fast):")
    start = time.perf_counter()
    action, level = decision_loop.decide(state, time_budget=0.005)  # 5ms
    elapsed = (time.perf_counter() - start) * 1000
    print(f"      Time budget: 5ms")
    print(f"      Actual time: {elapsed:.2f}ms")
    print(f"      Level used: {level.name}")
    print(f"      Action shape: {action.shape}")
    
    # Deliberative decision
    print("\n   b) Deliberative Loop (planning):")
    start = time.perf_counter()
    action, level = decision_loop.decide(state, time_budget=0.05)  # 50ms
    elapsed = (time.perf_counter() - start) * 1000
    print(f"      Time budget: 50ms")
    print(f"      Actual time: {elapsed:.2f}ms")
    print(f"      Level used: {level.name}")
    print(f"      Action shape: {action.shape}")
    
    # Meta-learning decision
    print("\n   c) Meta-Learning Loop (adaptive):")
    # Provide some trajectory context
    trajectories = [
        {
            "state": torch.randn(state_dim),
            "action": torch.randn(action_dim),
            "reward": torch.tensor([0.5])
        }
        for _ in range(5)
    ]
    context = {"trajectories": trajectories}
    
    start = time.perf_counter()
    action, level = decision_loop.decide(state, time_budget=0.5, context=context)  # 500ms
    elapsed = (time.perf_counter() - start) * 1000
    print(f"      Time budget: 500ms")
    print(f"      Actual time: {elapsed:.2f}ms")
    print(f"      Level used: {level.name}")
    print(f"      Action shape: {action.shape}")
    
    # Simulate learning updates
    print("\n4. Training decision loops...")
    for i in range(5):
        # Create synthetic experience
        states = torch.randn(10, state_dim)
        actions = torch.randn(10, action_dim)
        advantages = torch.randn(10)
        
        experience = {
            "states": states,
            "actions": actions,
            "advantages": advantages,
            "task_trajectories": []  # For meta-learning
        }
        
        metrics = decision_loop.update_all(experience)
        
        if (i + 1) % 2 == 0:
            print(f"   Update {i+1}: Reflex loss = {metrics.get('reflex_loss', 0.0):.4f}")
    
    # Compare latencies
    print("\n5. Latency comparison (100 iterations)...")
    
    latencies = {level.name: [] for level in DecisionLevel}
    
    for level_enum in DecisionLevel:
        if level_enum == DecisionLevel.REFLEX:
            budget = 0.005
        elif level_enum == DecisionLevel.DELIBERATIVE:
            budget = 0.05
        else:
            budget = 0.5
        
        for _ in range(100):
            start = time.perf_counter()
            action, used_level = decision_loop.decide(state, time_budget=budget)
            elapsed = (time.perf_counter() - start) * 1000
            if used_level == level_enum:
                latencies[level_enum.name].append(elapsed)
    
    print("\n   Latency statistics:")
    for level_name, times in latencies.items():
        if times:
            print(f"   {level_name}:")
            print(f"      Mean: {sum(times)/len(times):.2f}ms")
            print(f"      Min:  {min(times):.2f}ms")
            print(f"      Max:  {max(times):.2f}ms")
    
    print("\n" + "=" * 60)
    print("Hierarchical decision loops example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
