"""CartPole environment wrapper.

Built on `gymnasium`, the maintained successor to `gym`. The pinned `gym==0.21.0`
cannot be installed on current Python — its setup.py breaks against modern
setuptools — so `pip install -r requirements.txt` failed before anything ran.

Gymnasium's API differs from gym's in two ways that matter here: `reset()`
returns `(observation, info)` rather than a bare observation, and `step()`
returns five values, splitting the old `done` into `terminated` (the pole fell)
and `truncated` (the time limit was reached). Bootstrapping past a truncation is
correct and past a termination is not, so the two are kept distinct here and
combined only by the caller.
"""
import gymnasium as gym
import numpy as np


class CartPoleEnv:
    """Wrapper for the CartPole environment."""

    def __init__(self, render_mode=None):
        # gymnasium fixes the render mode at construction time
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0]   # 4 for CartPole
        self.action_dim = self.env.action_space.n              # 2 for CartPole

    def reset(self, seed=None):
        """Reset and return the preprocessed observation."""
        state, _info = self.env.reset(seed=seed)
        return self._preprocess_state(state)

    def step(self, action):
        """Take an action.

        Returns `(state, reward, terminated, truncated, info)`.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess_state(next_state), reward, terminated, truncated, info

    def _preprocess_state(self, state):
        """Normalise the state (a no-op for CartPole, kept for generality)."""
        return np.array(state, dtype=np.float32)

    def render(self):
        """Render the environment. Requires render_mode set at construction."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()
