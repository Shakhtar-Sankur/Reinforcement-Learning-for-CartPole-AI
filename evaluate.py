import gym
import torch
import numpy as np
from environment import CartPoleEnv
from agent import DQNAgent
from config import AGENT_PARAMS, TRAINING_PARAMS

def evaluate(model_path, num_episodes=100):
    """Evaluate the trained agent and report average reward."""
    env = CartPoleEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, **AGENT_PARAMS)
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.epsilon = 0.0  # Disable exploration
    
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    
    env.close()
    avg_reward = np.mean(rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward