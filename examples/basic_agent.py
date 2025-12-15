"""Example: Basic agent usage with a simple environment."""

import torch
import numpy as np
from rtdi import ModularAgent


class SimpleEnvironment:
    """Simple environment for demonstration."""

    def __init__(self, state_dim=10, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = None
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """Reset environment."""
        self.state = np.random.randn(self.state_dim)
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        """Take environment step."""
        # Simple dynamics: next_state = state + 0.1 * action + noise
        action_effect = action[:self.state_dim] if len(action) >= self.state_dim else np.pad(
            action, (0, self.state_dim - len(action))
        )
        self.state = self.state + 0.1 * action_effect + 0.05 * np.random.randn(self.state_dim)
        
        # Reward: negative distance from origin
        reward = -np.linalg.norm(self.state)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or reward > -0.5
        
        return self.state.copy(), reward, done, {}


def main():
    """Run basic agent example."""
    print("=" * 60)
    print("Real-Time Decision Intelligence - Basic Agent Example")
    print("=" * 60)
    
    # Configuration
    state_dim = 10
    action_dim = 4
    num_episodes = 10
    
    # Create agent
    print("\n1. Creating modular agent...")
    agent = ModularAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={
            "world_model": {
                "ensemble_size": 3,
                "hidden_dim": 128,
                "num_layers": 2
            },
            "decision_loop": {
                "planning_horizon": 5,
                "num_candidates": 5
            },
            "learner": {
                "replay_capacity": 10000
            }
        }
    )
    print(f"   Agent created with {state_dim}-dim state, {action_dim}-dim action")
    
    # Create environment
    env = SimpleEnvironment(state_dim, action_dim)
    
    # Training loop
    print(f"\n2. Training for {num_episodes} episodes...")
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from agent (using reflex loop for speed)
            action_tensor = agent.act(state_tensor, time_budget=0.01)
            action = action_tensor.squeeze(0).cpu().numpy()
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Learn online
            experience = {
                "state": state_tensor.squeeze(0),
                "action": action_tensor.squeeze(0),
                "reward": torch.tensor([reward]),
                "next_state": torch.FloatTensor(next_state),
                "done": torch.tensor([done])
            }
            metrics = agent.learn(experience)
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 2 == 0:
            avg_reward = np.mean(episode_rewards[-2:])
            print(f"   Episode {episode + 1}/{num_episodes}: "
                  f"Steps={steps}, Reward={episode_reward:.2f}, "
                  f"Avg(last 2)={avg_reward:.2f}")
    
    # Offline training
    print("\n3. Performing offline training...")
    history = agent.train_offline(num_epochs=5)
    if history.get("train_loss"):
        print(f"   Final training loss: {history['train_loss'][-1]:.4f}")
    
    # Get statistics
    print("\n4. Agent statistics:")
    stats = agent.get_stats()
    print(f"   Total steps: {stats['step_count']}")
    print(f"   Replay buffer size: {stats['learner_stats']['buffer_size']}")
    
    # Save agent
    checkpoint_path = "/tmp/agent_checkpoint.pt"
    agent.save(checkpoint_path)
    print(f"\n5. Agent saved to {checkpoint_path}")
    
    # Final evaluation
    print("\n6. Final evaluation (3 episodes)...")
    eval_rewards = []
    for _ in range(3):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = agent.act(state_tensor, time_budget=0.05)
            action = action_tensor.squeeze(0).cpu().numpy()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    print(f"   Evaluation reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
