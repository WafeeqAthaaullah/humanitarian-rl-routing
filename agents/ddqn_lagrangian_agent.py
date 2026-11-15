import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.dubai_humanitarian_env import DubaiHumanitarianEnv
import config


class DDQNNetwork(nn.Module):
    """11-dim state -> FC 128 -> ReLU -> FC 128 -> ReLU -> 5-dim Q-values"""

    def __init__(self, state_dim: int = 11, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


def _extract_action_mask(env, info, action_dim: int = 5) -> np.ndarray:
    if isinstance(info, dict):
        for key in ("action_mask", "valid_actions", "legal_actions", "mask"):
            if key in info and info[key] is not None:
                return np.asarray(info[key], dtype=bool)
    for attr in ("get_action_mask", "get_valid_actions", "action_mask"):
        if hasattr(env, attr):
            val = getattr(env, attr)
            mask = val() if callable(val) else val
            return np.asarray(mask, dtype=bool)
    return np.ones(action_dim, dtype=bool)


def _check_tier1_violation(info) -> bool:
    if not isinstance(info, dict):
        return False
    for key in ("tier1_violation", "tier_1_violation", "deadline_violated", "violation"):
        if bool(info.get(key, False)):
            return True
    return False


class DDQNLagrangianAgent:
    """Double DQN Agent with a Static Safety Penalty (-50.0) for Tier-1 violations."""

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.online_net = DDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    def act(self, state: np.ndarray, action_mask=None, epsilon: float = 0.0, **kwargs) -> int:
        # Handle callers that pass epsilon as the second positional argument: act(state, 0.05)
        if isinstance(action_mask, (float, int)):
            epsilon = float(action_mask)
            action_mask = kwargs.get("mask", None)

        if action_mask is None:
            action_mask = np.ones(self.action_dim, dtype=bool)
        else:
            action_mask = np.asarray(action_mask, dtype=bool)

        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0

        if random.random() < epsilon:
            return int(random.choice(valid_actions))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t).squeeze(0).cpu().numpy()

        masked_q = np.where(action_mask, q_values, -np.inf)
        return int(np.argmax(masked_q))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_masks_t = torch.BoolTensor(next_masks).to(self.device)

        current_q = self.online_net(states_t).gather(1, actions_t)

        # Double DQN Target:
        # 1. Select best valid next action using ONLINE network
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            next_q_online = next_q_online.masked_fill(~next_masks_t, float("-inf"))
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # 2. Evaluate that action's value using TARGET network
            next_q_target = self.target_net(next_states_t).gather(1, best_next_actions)
            target_q = rewards_t + self.gamma * next_q_target * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.online_net.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.online_net.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.online_net.load_state_dict(checkpoint)
        self.target_net.load_state_dict(self.online_net.state_dict())


def train_ddqn(
    num_episodes: int = 4000,
    save_path: str = "results/checkpoints/ddqn_lagrangian_model.pth",
):
    try:
        env = DubaiHumanitarianEnv()
    except TypeError:
        env = DubaiHumanitarianEnv(config)

    agent = DDQNLagrangianAgent(
        state_dim=11,
        action_dim=5,
        gamma=getattr(config, "GAMMA", 0.99),
        lr=getattr(config, "LR", 1e-3),
        batch_size=getattr(config, "BATCH_SIZE", 64),
    )

    eps_start = 1.0
    eps_end = 0.05
    eps_decay_episodes = int(num_episodes * 0.75)

    for episode in range(1, num_episodes + 1):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state, info = reset_out
        else:
            state, info = reset_out, {}

        action_mask = _extract_action_mask(env, info, agent.action_dim)
        done = False
        episode_reward = 0.0

        epsilon = max(
            eps_end,
            eps_start - (eps_start - eps_end) * (episode / eps_decay_episodes),
        )

        while not done:
            action = agent.act(state, action_mask=action_mask, epsilon=epsilon)
            step_out = env.step(action)

            if len(step_out) == 5:
                next_state, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_out

            next_mask = _extract_action_mask(env, info, agent.action_dim)

            # Static Safety Penalty (-50.0) when a Tier-1 cargo deadline is missed
            if _check_tier1_violation(info):
                reward -= 50.0

            agent.replay_buffer.push(state, action, reward, next_state, done, next_mask)
            agent.train_step()

            state = next_state
            action_mask = next_mask
            episode_reward += reward

        if episode % 250 == 0:
            print(
                f"Episode {episode}/{num_episodes} | Epsilon: {epsilon:.3f} | Reward: {episode_reward:.2f}"
            )

    agent.save(save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    train_ddqn(num_episodes=4000)