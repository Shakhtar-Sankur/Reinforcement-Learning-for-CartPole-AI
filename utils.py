import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def save_model(agent, path):
    """Save the agent's Q-network to the specified path."""
    torch.save(agent.q_network.state_dict(), path)

def log_metrics(rewards, episode, save_path):
    """Log rewards and save a plot of rewards vs. episodes."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Progress (Episode {episode})')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
    # Save rewards to CSV
    with open('plots/rewards.csv', 'a') as f:
        f.write(f"{episode},{rewards[-1]}\n")