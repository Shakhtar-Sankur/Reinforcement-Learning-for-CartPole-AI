"""The DQN agent: shapes, exploration, the replay buffer and one learning step.

These check behaviour that has a right answer — a greedy agent picks the action
with the highest Q-value, epsilon decays once per episode and stops at its floor,
the buffer is bounded, and a gradient step moves the network towards a target it
is asked to fit — rather than that training "runs".
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agent import DQNAgent


def make_agent(**overrides):
    settings = dict(state_dim=4, action_dim=2, learning_rate=1e-3, discount_factor=0.99,
                    epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
    settings.update(overrides)
    return DQNAgent(**settings)


def test_the_networks_match_the_environment_shape():
    agent = make_agent()
    q_values = agent.q_network(torch.zeros(1, 4))
    assert q_values.shape == (1, 2)


def test_target_network_starts_identical_to_the_q_network():
    agent = make_agent()
    for a, b in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        assert torch.equal(a, b)


def test_a_greedy_agent_picks_the_highest_q_value():
    agent = make_agent(epsilon_start=0.0)
    with torch.no_grad():                      # force a known preference for action 1
        agent.q_network[-1].bias.copy_(torch.tensor([-5.0, 5.0]))
    assert agent.select_action(np.zeros(4, dtype=np.float32)) == 1

    with torch.no_grad():
        agent.q_network[-1].bias.copy_(torch.tensor([5.0, -5.0]))
    assert agent.select_action(np.zeros(4, dtype=np.float32)) == 0


def test_a_fully_exploring_agent_uses_both_actions():
    agent = make_agent(epsilon_start=1.0)
    random.seed(0)
    chosen = {agent.select_action(np.zeros(4, dtype=np.float32)) for _ in range(50)}
    assert chosen == {0, 1}


def test_actions_are_always_legal():
    agent = make_agent(epsilon_start=0.5)
    random.seed(1)
    assert all(a in (0, 1) for a in (agent.select_action(np.random.randn(4).astype(np.float32))
                                     for _ in range(100)))


def test_epsilon_decays_per_call_and_stops_at_the_floor():
    agent = make_agent(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    for _ in range(50):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1), "epsilon must not fall below its floor"


def test_learning_does_not_change_epsilon():
    """Decay belongs to the episode loop; update() runs every step.

    This is the defect the README records: decaying inside update() emptied the
    exploration schedule in about three episodes.
    """
    agent = make_agent(epsilon_start=1.0)
    fill_buffer(agent, 200)
    agent.update()
    assert agent.epsilon == 1.0


def fill_buffer(agent, n, reward=1.0):
    for _ in range(n):
        agent.store_transition(np.random.randn(4).astype(np.float32),
                               random.randrange(2), reward,
                               np.random.randn(4).astype(np.float32), False)


def test_the_replay_buffer_is_bounded():
    agent = make_agent()
    fill_buffer(agent, 10_500)
    assert len(agent.memory) == 10_000, "the buffer must forget its oldest transitions"


def test_update_is_a_no_op_until_there_is_a_batch():
    agent = make_agent()
    fill_buffer(agent, agent.batch_size - 1)
    before = [p.clone() for p in agent.q_network.parameters()]
    agent.update()
    assert all(torch.equal(a, b) for a, b in zip(before, agent.q_network.parameters()))


def test_one_update_moves_the_network_and_only_the_online_one():
    agent = make_agent(learning_rate=0.01)
    fill_buffer(agent, 256)
    online_before = [p.clone() for p in agent.q_network.parameters()]
    target_before = [p.clone() for p in agent.target_network.parameters()]

    agent.update()

    assert any(not torch.equal(a, b) for a, b in zip(online_before, agent.q_network.parameters())), \
        "a gradient step changed nothing"
    assert all(torch.equal(a, b) for a, b in zip(target_before, agent.target_network.parameters())), \
        "the target network must only change when it is copied"


def test_repeated_updates_reduce_the_error_on_a_fixed_batch():
    """The one thing that matters: the network learns what it is shown.

    Every transition is terminal with reward 1, so every target is exactly 1.
    After a few hundred steps on that batch the Q-values should be near 1.
    """
    torch.manual_seed(0)
    random.seed(0)
    agent = make_agent(learning_rate=0.01, epsilon_start=0.0)
    states = np.random.randn(64, 4).astype(np.float32)
    for index in range(64):
        agent.store_transition(states[index], index % 2, 1.0, states[index], True)

    def mean_q():
        with torch.no_grad():
            return float(agent.q_network(torch.FloatTensor(states)).mean())

    start_error = abs(mean_q() - 1.0)
    for _ in range(300):
        agent.update()
    end_error = abs(mean_q() - 1.0)

    assert end_error < start_error, "training did not reduce the error at all"
    assert end_error < 0.25, f"after 300 steps the Q-values are still {end_error:.2f} from the target"


def test_copying_to_the_target_network_makes_them_equal_again():
    agent = make_agent(learning_rate=0.05)
    fill_buffer(agent, 200)
    agent.update()
    agent.update_target_network()
    for a, b in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        assert torch.equal(a, b)
