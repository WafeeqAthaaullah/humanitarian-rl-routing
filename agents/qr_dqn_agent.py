import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from config import (
    OBS_LOW, OBS_HIGH,
    STATE_DIM, ACTION_DIM, NUM_QUANTILES, HIDDEN_DIM, CVAR_ALPHA,
    LEARNING_RATE, GAMMA, BATCH_SIZE, BUFFER_CAPACITY, GRAD_CLIP_NORM,
)


class QRDQNNetwork(nn.Module):
    """
    Unlike a standard DQN that outputs a single average score per action,
    QR-DQN outputs a full distribution (N quantiles) for every action.
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, num_quantiles=NUM_QUANTILES):
        super(QRDQNNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        
        # A simple feed-forward neural network
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            # Output size: action_dim * num_quantiles (e.g. 5 * 50 = 250)
            nn.Linear(HIDDEN_DIM, action_dim * num_quantiles) 
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.feature_layer(x)
        # Reshape to [batch_size, action_dim, num_quantiles]
        return out.view(batch_size, self.action_dim, self.num_quantiles)

class QRDQNAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, num_quantiles=NUM_QUANTILES, alpha=CVAR_ALPHA):
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.alpha = alpha # CVaR threshold (worst 10%)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = QRDQNNetwork(state_dim, action_dim, num_quantiles).to(self.device)

        # Pre-compute normalization tensors for observation scaling
        self._obs_low = torch.FloatTensor(OBS_LOW).to(self.device)
        obs_range = OBS_HIGH - OBS_LOW
        # Clamp range to avoid division-by-zero for any constant-valued features
        self._obs_range = torch.FloatTensor(np.maximum(obs_range, 1e-6)).to(self.device)

    def _normalize(self, state_tensor):
        """Scale observations to roughly [0, 1] for stable neural-network training.

        Features with wildly different scales (e.g. volume 1–100 vs capacity 0–1)
        cause uneven gradient magnitudes.  This simple min-max normalization using
        the declared observation-space bounds eliminates the problem.
        """
        return (state_tensor - self._obs_low) / self._obs_range

    def act_cvar(self, state, epsilon=0.0, action_mask=None):
        """
        Risk-sensitive action selection (CVaR).
        Looks at the worst 10% of predicted outcomes and chooses the safest route.
        """
        # Epsilon-greedy exploration
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
            quantiles = self.network(state_tensor) # Shape: [1, 5, 50]
            
            # Sort quantiles to find the worst-case outcomes (lowest returns)
            sorted_quantiles, _ = torch.sort(quantiles, dim=2)
            
            # Select the bottom alpha fraction (e.g., the worst 5 out of 50 quantiles)
            cutoff_index = max(1, int(self.alpha * self.num_quantiles))
            worst_case_returns = sorted_quantiles[:, :, :cutoff_index]
            
            # Calculate the mean of these worst-case outcomes for each action
            cvar_values = worst_case_returns.mean(dim=2).squeeze(0)
            cvar_values[~torch.as_tensor(action_mask, device=self.device)] = -torch.inf
            
            # Pick the action with the "least bad" worst-case outcome
            return torch.argmax(cvar_values).item()
        
    def setup_training(self, lr=LEARNING_RATE, gamma=GAMMA):
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity=BUFFER_CAPACITY)
        
        # Target network for stable learning
        self.target_network = QRDQNNetwork(STATE_DIM, self.action_dim, self.num_quantiles).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Calculate quantile midpoints (tau) for the loss function
        tau = torch.Tensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)).to(self.device)
        self.tau = tau.view(1, -1)

    def train_step(self, batch_size=BATCH_SIZE):
        if len(self.memory) < batch_size:
            return 0.0 # Not enough experience to train yet
            
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(batch_size)
        
        # Normalize observations before feeding to the network
        states = self._normalize(torch.FloatTensor(states).to(self.device))
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_states = self._normalize(torch.FloatTensor(next_states).to(self.device))
        next_masks = torch.BoolTensor(next_masks).to(self.device)
        
        # 1. Get current quantile predictions for the actions taken
        current_quantiles = self.network(states) # [batch, actions, quantiles]
        current_action_quantiles = current_quantiles.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.num_quantiles)).squeeze(1)
        
        # 2. Get target quantiles from the target network
        with torch.no_grad():
            next_quantiles = self.network(next_states)
            # Greedy action selection based on mean return
            next_values = next_quantiles.mean(dim=2)
            next_values = next_values.masked_fill(~next_masks, -torch.inf)
            next_actions = next_values.argmax(dim=1, keepdim=True)
            
            target_next_quantiles = self.target_network(next_states)
            target_action_quantiles = target_next_quantiles.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, self.num_quantiles)).squeeze(1)
            
            # Bellman equation applied to quantiles
            target_quantiles = rewards + self.gamma * (1 - dones) * target_action_quantiles

        # 3. Quantile Huber Loss calculation
        current_expanded = current_action_quantiles.unsqueeze(2).expand(-1, -1, self.num_quantiles)
        target_expanded = target_quantiles.unsqueeze(1).expand(-1, self.num_quantiles, -1)
        
        diff = target_expanded - current_expanded
        huber_loss = F.smooth_l1_loss(current_expanded, target_expanded, reduction='none')
        
        loss = (torch.abs(self.tau.unsqueeze(0) - (diff < 0).float()) * huber_loss).sum(dim=2).mean()
        
        # 4. Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_action_mask=None):
        if next_action_mask is None:
            next_action_mask = np.ones(5, dtype=bool)
        self.buffer.append((state, action, reward, next_state, done, next_action_mask))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), 
                np.array(next_states), np.array(dones, dtype=np.float32),
                np.array(next_masks, dtype=bool))
    
    def __len__(self):
        return len(self.buffer)