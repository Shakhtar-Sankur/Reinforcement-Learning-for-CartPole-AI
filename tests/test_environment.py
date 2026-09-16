"""The environment wrapper and the small helpers around training.

The wrapper's whole job is to keep gymnasium's `terminated` and `truncated`
apart — bootstrapping past a time limit is correct, past a fallen pole is not —
so that is what these check, along with the saved model being loadable.
"""

import csv

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

from environment import CartPoleEnv


@pytest.fixture
def env():
    e = CartPoleEnv()
    yield e
    e.close()


def test_the_wrapper_reports_cartpoles_real_dimensions(env):
    assert env.state_dim == 4
    assert env.action_dim == 2


def test_reset_returns_one_observation_not_gymnasiums_tuple(env):
    state = env.reset(seed=0)
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)
    assert state.dtype == np.float32


def test_the_same_seed_gives_the_same_start(env):
    assert np.array_equal(env.reset(seed=123), env.reset(seed=123))


def test_step_keeps_terminated_and_truncated_separate(env):
    env.reset(seed=0)
    state, reward, terminated, truncated, info = env.step(0)
    assert state.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_an_episode_ends_and_says_which_way_it_ended(env):
    """Pushing one way only makes the pole fall: terminated, not truncated."""
    env.reset(seed=0)
    for step in range(500):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            assert terminated and not truncated, "falling over is a termination, not a time limit"
            assert step < 100, "the pole should fall quickly under a constant push"
            return
    pytest.fail("the episode never ended")


def test_a_full_episode_can_be_played_with_the_agent():
    """The two pieces fit together: agent chooses, environment steps."""
    from agent import DQNAgent

    env = CartPoleEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, learning_rate=1e-3,
                     discount_factor=0.99, epsilon_start=0.2, epsilon_end=0.01, epsilon_decay=0.99)
    state = env.reset(seed=1)
    total = 0.0
    for _ in range(200):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, terminated)
        agent.update()
        total += reward
        state = next_state
        if terminated or truncated:
            break
    env.close()
    assert total >= 1, "no reward collected at all"
    assert len(agent.memory) > 0


def test_a_saved_model_can_be_loaded_back(tmp_path):
    from agent import DQNAgent
    from utils import save_model

    agent = DQNAgent(state_dim=4, action_dim=2, learning_rate=1e-3, discount_factor=0.99,
                     epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.99)
    path = tmp_path / "nested" / "model.pth"      # a directory that does not exist yet

    save_model(agent, str(path))

    assert path.exists(), "save_model did not create the directory it was given"
    loaded = torch.load(str(path), map_location="cpu")
    fresh = DQNAgent(state_dim=4, action_dim=2, learning_rate=1e-3, discount_factor=0.99,
                     epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.99)
    fresh.q_network.load_state_dict(loaded)
    for a, b in zip(agent.q_network.parameters(), fresh.q_network.parameters()):
        assert torch.equal(a, b)


def test_metrics_are_written_beside_the_plot_with_a_header(tmp_path):
    """The CSV goes next to the plot it belongs to, not a hard-coded folder."""
    pytest.importorskip("matplotlib")
    from utils import log_metrics

    plot = tmp_path / "runs" / "progress.png"
    log_metrics([10.0], episode=1, save_path=str(plot))
    log_metrics([10.0, 25.0], episode=2, save_path=str(plot))

    csv_path = tmp_path / "runs" / "rewards.csv"
    assert plot.exists() and csv_path.exists()
    rows = list(csv.reader(csv_path.open()))
    assert rows[0] == ["episode", "reward"], "header written once, at the top"
    assert rows[1][0] == "1" and rows[2][0] == "2"
    assert float(rows[2][1]) == 25.0, "the latest reward, not the whole list"
