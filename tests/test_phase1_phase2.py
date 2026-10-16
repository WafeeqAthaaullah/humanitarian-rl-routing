import numpy as np
import torch

from agents.qr_dqn_agent import QRDQNAgent
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv


def test_environment_exposes_legal_action_mask_and_metrics():
    env = DubaiHumanitarianEnv()
    state, reset_info = env.reset(seed=7)

    assert reset_info["action_mask"].tolist() == [True, True, True, False, False]
    action = 1
    _, _, _, _, info = env.step(action)

    assert "freight_cost" in info
    assert "deadline_slack" in info
    assert "tier1_violation" in info
    assert info["action_mask"].tolist() == [False, False, False, True, True]


def test_cvar_exploration_respects_action_mask():
    env = DubaiHumanitarianEnv()
    state, info = env.reset(seed=11)
    agent = QRDQNAgent()

    actions = [
        agent.act_cvar(state, epsilon=1.0, action_mask=info["action_mask"])
        for _ in range(50)
    ]

    assert all(info["action_mask"][action] for action in actions)


def test_replay_buffer_preserves_next_action_masks():
    agent = QRDQNAgent()
    agent.setup_training()
    state = np.zeros(11, dtype=np.float32)
    mask = np.array([True, False, True, False, False])
    agent.memory.push(state, 0, -1.0, state, True, mask)

    _, _, _, _, _, sampled_masks = agent.memory.sample(1)

    assert sampled_masks.dtype == bool
    assert sampled_masks[0].tolist() == mask.tolist()
