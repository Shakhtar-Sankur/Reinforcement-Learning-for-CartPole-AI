# Reinforcement Learning — CartPole

**A DQN agent for CartPole-v1, written out in full: replay buffer, target network,
epsilon schedule, evaluation and a small Streamlit viewer.**

CartPole is a teaching problem, and this repository treats it as one. The point is a
readable, complete DQN rather than a competitive result.

## What's here

| File | Role |
|---|---|
| `agent.py` | DQN agent — Q-network, replay memory, epsilon-greedy action selection |
| `environment.py` | Gymnasium wrapper, keeping `terminated` and `truncated` distinct |
| `main.py` | Training loop |
| `evaluate.py` | Runs a trained policy without exploration |
| `config.py` | `AGENT_PARAMS` and `TRAINING_PARAMS` |
| `utils.py` | Model checkpointing, reward logging and plots |
| `visualize.py` | Plot rendering |
| `app.py` | Streamlit front end |

## Design target

Balance the pole for the full 500-step episode consistently, which is the standard
solved threshold for CartPole-v1.

## On the numbers

The figures above are **design targets** that shaped the implementation — they are not
measured results. This repository ships no benchmark harness and no trained weights, so
nothing here reproduces them. They are recorded because they drove real decisions about
architecture and algorithm choice, not as claims about observed performance.

## Running it

```bash
pip install -r requirements.txt
python main.py           # train
python evaluate.py       # evaluate a saved policy
streamlit run app.py
```

Plots and `rewards.csv` are written next to each other, and the output directory is created
automatically.

A short run to show it learns — 60 episodes, which is well short of solving:

```
Episode  0, Reward:  17.00
Episode 20, Reward:  66.00
Episode 40, Reward: 106.00

first 10 mean : 20.3
last 10 mean  : 26.0
best episode  : 106
```

Still erratic at that point, because epsilon is around 0.74 after 60 episodes and most
actions are still random. CartPole-v1 counts as solved at an average of 475 over 100
consecutive episodes; the default configuration runs 500.

## Status

Complete and runnable.

### Notes from a correctness pass

Three defects fixed, the first of which stopped everything:

- **`config.py` was a byte-identical copy of `app.py`** — the Streamlit demo — and defined
  no hyperparameters at all. `main.py`, `evaluate.py` and `visualize.py` all do
  `from config import HYPERPARAMS`, so every one of them failed at import. There was also a
  latent conflict: `HYPERPARAMS` was splatted into `DQNAgent(...)` *and* read for
  `num_episodes`, which the agent does not accept. Agent and training settings are now
  separate dictionaries.
- **Exploration collapsed almost immediately.** Epsilon decayed inside `update()`, which
  runs on every environment step. At roughly 200 steps an episode, a 0.995 factor reaches
  the floor in about three episodes, so the agent stopped exploring before it had seen
  anything. Decay is now an explicit `decay_epsilon()` called once per episode.
- **`gym==0.21.0` cannot be installed on current Python** — its `setup.py` breaks against
  modern setuptools — so `pip install -r requirements.txt` failed before any of this
  mattered. Now `gymnasium`, with `terminated` and `truncated` kept distinct, since
  bootstrapping past a time limit is correct and bootstrapping past a real termination is
  not.

The target network was also being refreshed every 100 episodes, which is only a handful of
updates across a run; it is every 10 now.

## Licence

All rights reserved. Published for reading, not for reuse.
