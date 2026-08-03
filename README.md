# Reinforcement Learning — CartPole

**A DQN agent for CartPole-v1, written out in full: replay buffer, target network,
epsilon schedule, evaluation and a small Streamlit viewer.**

CartPole is a teaching problem, and this repository treats it as one. The point is a
readable, complete DQN rather than a competitive result.

## What's here

| File | Role |
|---|---|
| `agent.py` | DQN agent — Q-network, replay memory, epsilon-greedy action selection |
| `environment.py` | Gym environment wrapper |
| `main.py` | Training loop |
| `evaluate.py` | Runs a trained policy without exploration |
| `config.py` | Hyperparameters in one place |
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

Plots and `rewards.csv` are written next to each other, and the output directory is
created automatically.

## Status

Complete and runnable.

## Licence

All rights reserved. Published for reading, not for reuse.
