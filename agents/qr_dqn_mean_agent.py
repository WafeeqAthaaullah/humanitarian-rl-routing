import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.qr_dqn_agent import QRDQNNetwork, ReplayBuffer
from config import (
    ACTION_DIM,
    BATCH_SIZE,
    BUFFER_CAPACITY,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_AIR_CAPACITY,
    MAX_LAND_CAPACITY,
    MAX_OCEAN_CAPACITY,
    NUM_QUANTILES,
    OBS_HIGH,
    OBS_LOW,
    STATE_DIM,
    TARGET_UPDATE_FREQ,
)
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv

MEAN_TRAIN_EPISODES = 4000
MODEL_SAVE_PATH = "results/checkpoints/qr_dqn_mean_model.pth"


class QRDQNMeanAgent:
    """
    Risk-Neutral Distributional QR-DQN Agent (Mean Action Selection).
    Trains on 50 quantiles and selects actions using the mean expected return
    across all 50 quantiles rather than CVaR tail risk.
    """

    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        num_quantiles=NUM_QUANTILES,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = QRDQNNetwork(state_dim, action_dim, num_quantiles).to(self.device)

        # Pre-compute normalization tensors for observation scaling
        self._obs_low = torch.FloatTensor(OBS_LOW).to(self.device)
        obs_range = OBS_HIGH - OBS_LOW
        self._obs_range = torch.FloatTensor(np.maximum(obs_range, 1e-6)).to(self.device)

    def _normalize(self, state_tensor):
        """Scale observations to [0, 1] using observation-space bounds."""
        return (state_tensor - self._obs_low) / self._obs_range

    def _infer_action_mask(self, state):
        """Infer legal action mask from state if action_mask is not explicitly passed."""
        location = state[4]
        weight = state[2]
        capacities = state[6:9]
        required = np.array(
            [
                weight / MAX_AIR_CAPACITY,
                weight / MAX_OCEAN_CAPACITY,
                weight / MAX_LAND_CAPACITY,
                weight / MAX_AIR_CAPACITY,
                0.0,
            ]
        )
        mask = capacities[[0, 1, 2, 0, 0]] >= required
        if location == 0.0:
            mask[[3, 4]] = False
        elif location == 1.0:
            mask[[0, 1, 2]] = False
        else:
            mask[:] = False
        return mask.astype(bool)

    def act_mean(self, state, epsilon=0.0, action_mask=None):
        """
        Risk-neutral action selection using Mean Expected Return:
            Q_mean(state, action) = Average of all 50 predicted quantiles for action
            Selects argmax_a(Q_mean(state, action)) among valid actions.
        """
        if action_mask is None:
            action_mask = self._infer_action_mask(state)
        action_mask = np.asarray(action_mask, dtype=bool)
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            raise ValueError("Action mask contains no valid actions")

        # Epsilon-greedy exploration among valid actions
        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_tensor = self._normalize(state_tensor)

        with torch.no_grad():
            # Predict 50 quantiles for each of the 5 actions -> Shape: [1, 5, 50]
            quantiles = self.network(state_tensor)

            # Compute the mean value across all 50 quantiles for each action -> Shape: [5]
            q_mean = quantiles.mean(dim=2).squeeze(0)

            # Mask out invalid actions and pick argmax_a(Q_mean(state, action))
            q_mean[~torch.as_tensor(action_mask, device=self.device)] = -torch.inf
            return int(torch.argmax(q_mean).item())

    def act(self, state, epsilon=0.0, action_mask=None):
        """Alias for evaluation script compatibility."""
        return self.act_mean(state, epsilon=epsilon, action_mask=action_mask)

    def act_cvar(self, state, epsilon=0.0, action_mask=None):
        """Alias if evaluate_model.py calls act_cvar on all RL agents."""
        return self.act_mean(state, epsilon=epsilon, action_mask=action_mask)

    def setup_training(self, lr=LEARNING_RATE, gamma=GAMMA):
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity=BUFFER_CAPACITY)

        self.target_network = QRDQNNetwork(
            self.state_dim, self.action_dim, self.num_quantiles
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        tau = torch.Tensor(
            (2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)
        ).to(self.device)
        self.tau = tau.view(1, -1)

    def train_step(self, batch_size=BATCH_SIZE):
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(
            batch_size
        )

        states = self._normalize(torch.FloatTensor(states).to(self.device))
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_states = self._normalize(torch.FloatTensor(next_states).to(self.device))
        next_masks = torch.BoolTensor(next_masks).to(self.device)

        # 1. Current quantile predictions for taken actions
        current_quantiles = self.network(states)  # [batch, 5, 50]
        current_action_quantiles = current_quantiles.gather(
            1, actions.unsqueeze(-1).expand(-1, -1, self.num_quantiles)
        ).squeeze(1)

        # 2. Target quantiles using mean action selection
        with torch.no_grad():
            next_quantiles = self.network(next_states)
            next_values = next_quantiles.mean(dim=2)
            next_values = next_values.masked_fill(~next_masks, -torch.inf)
            next_actions = next_values.argmax(dim=1, keepdim=True)

            target_next_quantiles = self.target_network(next_states)
            target_action_quantiles = target_next_quantiles.gather(
                1, next_actions.unsqueeze(-1).expand(-1, -1, self.num_quantiles)
            ).squeeze(1)

            target_quantiles = (
                rewards + self.gamma * (1 - dones) * target_action_quantiles
            )

        # 3. Quantile Huber Loss
        current_expanded = current_action_quantiles.unsqueeze(2).expand(
            -1, -1, self.num_quantiles
        )
        target_expanded = target_quantiles.unsqueeze(1).expand(
            -1, self.num_quantiles, -1
        )

        diff = target_expanded - current_expanded
        huber_loss = F.smooth_l1_loss(
            current_expanded, target_expanded, reduction="none"
        )

        loss = (
            (torch.abs(self.tau.unsqueeze(0) - (diff < 0).float()) * huber_loss)
            .sum(dim=2)
            .mean()
        )

        # 4. Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), max_norm=GRAD_CLIP_NORM
        )
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())


def train_qr_dqn_mean(episodes=MEAN_TRAIN_EPISODES, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    env = DubaiHumanitarianEnv()
    agent = QRDQNMeanAgent()
    agent.setup_training(lr=LEARNING_RATE, gamma=GAMMA)

    epsilon = EPSILON_START
    violations_history = []

    print(f"Starting Risk-Neutral QR-DQN (Mean) training for {episodes} episodes...")

    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=seed + ep)
        is_tier1 = bool(state[0] == 1.0)
        episode_reward = 0.0
        loss_sum = 0.0
        steps = 0
        violated = False

        while True:
            action_mask = info.get("action_mask")
            action = agent.act_mean(state, epsilon=epsilon, action_mask=action_mask)

            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated and info.get("tier1_violation", False):
                violated = True

            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                terminated,
                info.get("action_mask"),
            )

            loss = agent.train_step(batch_size=BATCH_SIZE)
            loss_sum += loss

            state = next_state
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        violations_history.append(1 if violated else 0)

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if ep % 100 == 0:
            avg_loss = loss_sum / max(1, steps)
            recent_viol = np.mean(violations_history[-100:]) * 100
            print(
                f"Episode {ep:4d}/{episodes} | Reward: {episode_reward:6.2f} | "
                f"Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f} | "
                f"Violation Rate (100-ep): {recent_viol:.1f}%"
            )

    torch.save(agent.network.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model weights saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_qr_dqn_mean()
