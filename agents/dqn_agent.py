"""
Standard Deep Q-Network (DQN) baseline — risk-neutral scalar Q-learning.

Assigned to: AFRITH
Proves why scalar expectation fails under fat-tailed maritime disruption delays.
No quantile regression, no Lagrangian safety penalties.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import random
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import (
    OBS_LOW, OBS_HIGH,
    STATE_DIM, ACTION_DIM, HIDDEN_DIM,
    LEARNING_RATE, GAMMA, BATCH_SIZE, BUFFER_CAPACITY, GRAD_CLIP_NORM,
    EPSILON_START, EPSILON_DECAY, EPSILON_MIN, TARGET_UPDATE_FREQ,
)
from agents.qr_dqn_agent import ReplayBuffer
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv


class DQNNetwork(nn.Module):
    """
    Classic DQN: maps an 11-dim observation to 5 scalar Q-values.

        State (11) -> FC 128 -> ReLU -> FC 128 -> ReLU -> Q-Values (5)
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super(DQNNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.feature_layer(x)


class DQNAgent:
    """Risk-neutral Deep Q-Network with epsilon-greedy masked action selection."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = DQNNetwork(state_dim, action_dim).to(self.device)

        self._obs_low = torch.FloatTensor(OBS_LOW).to(self.device)
        obs_range = OBS_HIGH - OBS_LOW
        self._obs_range = torch.FloatTensor(np.maximum(obs_range, 1e-6)).to(self.device)

    def _normalize(self, state_tensor):
        """Min-max normalize observations using declared observation-space bounds."""
        return (state_tensor - self._obs_low) / self._obs_range

    def act(self, state, epsilon=0.0, action_mask=None):
        """
        Epsilon-greedy action selection among legal actions from DubaiHumanitarianEnv.
        """
        if action_mask is None:
            action_mask = np.ones(self.action_dim, dtype=bool)
        action_mask = np.asarray(action_mask, dtype=bool)
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            raise ValueError("Action mask contains no valid actions")

        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_tensor = self._normalize(state_tensor)

        with torch.no_grad():
            q_values = self.network(state_tensor).squeeze(0)
            q_values[~torch.as_tensor(action_mask, device=self.device)] = -torch.inf
            return int(torch.argmax(q_values).item())

    def setup_training(self, lr=LEARNING_RATE, gamma=GAMMA):
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity=BUFFER_CAPACITY)

        self.target_network = DQNNetwork(STATE_DIM, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

    def train_step(self, batch_size=BATCH_SIZE):
        """
        Standard DQN update with MSE loss against scalar Bellman targets:

            Target = Reward + Gamma * max_a' Q_target(next_state, a')
        """
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(batch_size)

        states = self._normalize(torch.FloatTensor(states).to(self.device))
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_states = self._normalize(torch.FloatTensor(next_states).to(self.device))
        next_masks = torch.BoolTensor(next_masks).to(self.device)

        current_q = self.network(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_q = next_q.masked_fill(~next_masks, -torch.inf)
            max_next_q = next_q.max(dim=1, keepdim=True).values
            # If every next action is masked (terminal), treat max Q as 0
            max_next_q = torch.where(
                torch.isfinite(max_next_q),
                max_next_q,
                torch.zeros_like(max_next_q),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())


def train_dqn(episodes=3000, seed=42, checkpoint_path="results/checkpoints/dqn_model.pth"):
    """Train the standard DQN baseline and save weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    env = DubaiHumanitarianEnv()
    agent = DQNAgent()
    agent.setup_training(lr=LEARNING_RATE)

    epsilon = EPSILON_START
    print(f"Starting standard DQN training for {episodes} episodes...")

    for ep in range(episodes):
        state, info = env.reset(seed=seed + ep)
        episode_reward = 0.0
        loss_sum = 0.0
        steps = 0

        while True:
            action = agent.act(state, epsilon, action_mask=info.get("action_mask"))
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.memory.push(
                state, action, reward, next_state, terminated,
                info.get("action_mask"),
            )
            loss_sum += agent.train_step(batch_size=BATCH_SIZE)

            state = next_state
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if ep % 100 == 0:
            avg_loss = loss_sum / max(1, steps)
            print(
                f"Episode {ep:4d} | Reward: {episode_reward:6.2f} | "
                f"Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}"
            )

    torch.save(
        {
            "model_state_dict": agent.network.state_dict(),
            "episode": episodes,
            "seed": seed,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "agent": "dqn",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        checkpoint_path,
    )
    print(f"Model saved to {checkpoint_path}")
    return agent


if __name__ == "__main__":
    train_dqn(episodes=3000)
