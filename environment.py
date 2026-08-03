import gym
import numpy as np

class CartPoleEnv:
    """Wrapper for the CartPole environment."""
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_dim = self.env.observation_space.shape[0]  # 4 for CartPole
        self.action_dim = self.env.action_space.n  # 2 for CartPole
    
    def reset(self):
        """Reset the environment and return normalized state."""
        state = self.env.reset()
        return self._preprocess_state(state)
    
    def step(self, action):
        """Take an action and return preprocessed state, reward, done, info."""
        next_state, reward, done, info = self.env.step(action)
        return self._preprocess_state(next_state), reward, done, info
    
    def _preprocess_state(self, state):
        """Normalize the state (optional for CartPole, included for generality)."""
        return np.array(state, dtype=np.float32)
    
    def render(self):
        """Render the environment."""
        self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()