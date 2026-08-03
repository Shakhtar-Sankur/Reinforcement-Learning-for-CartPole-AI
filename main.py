import torch
import numpy as np
from agent import DQNAgent
from environment import CartPoleEnv
from utils import save_model, log_metrics
from config import AGENT_PARAMS, TRAINING_PARAMS
import os

def train_dqn():
    """Train a DQN agent on the CartPole environment."""
    # Initialize environment and agent
    env = CartPoleEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, **AGENT_PARAMS)
    
    # Training parameters
    num_episodes = TRAINING_PARAMS['num_episodes']
    rewards = []
    target_update_freq = TRAINING_PARAMS['target_update_freq']
    
    try:
        # Training loop
        for episode in range(num_episodes):
            # Handle new Gym API (returns state, info)
            reset_result = env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                # Handle new Gym API (returns state, reward, done, truncated, info)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, done, truncated, _ = step_result
                    done = done or truncated  # Combine done and truncated for compatibility
                else:
                    next_state, reward, done, _ = step_result
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            agent.decay_epsilon()
            
            # Update target network
            if episode % target_update_freq == 0:
                agent.update_target_network()
            
            # Log and save progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}")
                try:
                    log_metrics(rewards, episode, save_path="plots/rewards.png")
                except Exception as e:
                    print(f"Warning: Failed to save plot: {e}")
            
            # Save model periodically
            if episode % 100 == 0:
                try:
                    save_model(agent, f"models/dqn_cartpole_{episode}.pth")
                except Exception as e:
                    print(f"Warning: Failed to save model: {e}")
        
    finally:
        env.close()
    
    return rewards

if __name__ == "__main__":
    # Create directories if they don't exist
    try:
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("videos", exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create directories: {e}")
    
    rewards = train_dqn()
    print(f"Training completed. Average reward: {np.mean(rewards[-100:]):.2f}")