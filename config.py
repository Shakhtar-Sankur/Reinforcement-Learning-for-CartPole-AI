"""Hyperparameters for the DQN agent and the training loop.

This file previously held a byte-identical copy of the Streamlit demo in
`app.py` and defined no hyperparameters at all, so `main.py`, `evaluate.py` and
`visualize.py` each failed at import.

The two dictionaries are separate on purpose. `AGENT_PARAMS` is splatted into
`DQNAgent(...)`, which accepts exactly these keyword arguments; putting a
training-loop setting such as `num_episodes` in there raises
`TypeError: unexpected keyword argument`.
"""

#: Passed straight to DQNAgent.
AGENT_PARAMS = {
    "learning_rate": 1e-3,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    # Applied once per episode, not per gradient step. At ~200 steps an episode a
    # per-step decay of 0.995 drives epsilon to its floor within about three
    # episodes, and the agent stops exploring before it has seen anything.
    "epsilon_decay": 0.995,
}

#: Used by the training loop.
TRAINING_PARAMS = {
    "num_episodes": 500,
    # CartPole-v1 is solved at an average of 475 over 100 consecutive episodes.
    "solved_threshold": 475.0,
    # Copying the online weights into the target network every 10 episodes keeps
    # the bootstrap target fresh; every 100, as this used to be, means only a
    # handful of updates across a whole run.
    "target_update_freq": 10,
    "log_every": 10,
    "checkpoint_every": 100,
}

#: Everything in one mapping, for callers that want a single view.
HYPERPARAMS = {**AGENT_PARAMS, **TRAINING_PARAMS}
