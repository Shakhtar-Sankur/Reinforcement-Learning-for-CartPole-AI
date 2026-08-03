import gym
import cv2
import numpy as np
from environment import CartPoleEnv
from agent import DQNAgent
from config import HYPERPARAMS

def record_video(model_path, video_path="videos/gameplay.mp4"):
    """Record a video of the trained agent playing CartPole."""
    env = CartPoleEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, **HYPERPARAMS)
    agent.q_network.load_state_dict(torch.load(model_path))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (600, 400))
    
    state = env.reset()
    done = False
    while not done:
        frame = env.env.render(mode='rgb_array')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
    
    out.release()
    env.close()