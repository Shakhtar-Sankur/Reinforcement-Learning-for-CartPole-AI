import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def save_model(agent, path):
    """Save the agent's Q-network to the specified path."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(agent.q_network.state_dict(), path)

def log_metrics(rewards, episode, save_path):
    """Log rewards and save a plot of rewards vs. episodes."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Progress (Episode {episode})')
    plt.legend()
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # Save rewards to CSV alongside the plot, rather than a hard-coded directory
    csv_path = os.path.join(os.path.dirname(save_path) or '.', 'rewards.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write("episode,reward\n")
        f.write(f"{episode},{rewards[-1]}\n")