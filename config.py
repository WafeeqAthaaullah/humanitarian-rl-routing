"""
Centralized configuration for the Humanitarian RL Routing project.
All environment constants, agent architecture parameters, and training
hyperparameters are defined here as a single source of truth.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

# Transport capacity limits (tons)
MAX_AIR_CAPACITY = 50.0
MAX_OCEAN_CAPACITY = 500.0
MAX_LAND_CAPACITY = 200.0

# Freight rates ($ per ton)
FREIGHT_RATES = {
    "air": 2000.0,
    "ocean": 200.0,
    "land": 500.0,
    "continue": 100.0,
}

# Base transit times (days)
BASE_TRANSIT_TIMES = {
    "air": 2.0,
    "ocean_leg1": 7.0,
    "land_leg1": 4.0,
    "ocean_leg2": 5.0,
    "land_leg2": 3.0,
}

# Mode-switch overhead
MODE_SWITCH_COST = 500.0       # additional dollar cost
MODE_SWITCH_DAYS = 1.0         # additional handling time (days)

# Reward / penalty tuning
INVALID_ACTION_PENALTY = -100.0
SAFETY_PENALTY = 50.0
COST_NORMALIZATION = 10000.0

# Transit-time noise parameters
TRANSIT_STD = 0.5
DISRUPTION_LOGNORM_MEAN = 2.0
DISRUPTION_LOGNORM_STD = 0.5
MIN_TRANSIT_TIME = 1.0

# Observation-space bounds
# [u, v, w, d_rem, loc, m_curr, c_air, c_ocean, c_land, i_dest, s_disrupt]
OBS_LOW = np.array(
    [1.0, 1.0, 0.5, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
OBS_HIGH = np.array(
    [2.0, 100.0, 50.0, 30.0, 2.0, 3.0, 1.0, 1.0, 1.0, 100.0, 1.0],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Agent architecture
# ---------------------------------------------------------------------------
STATE_DIM = 11
ACTION_DIM = 5
NUM_QUANTILES = 50
HIDDEN_DIM = 128
CVAR_ALPHA = 0.10   # worst-case fraction for CVaR action selection

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRAIN_EPISODES = 5000
LEARNING_RATE = 5e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
BATCH_SIZE = 32
BUFFER_CAPACITY = 50000
TARGET_UPDATE_FREQ = 10
LAMBDA_LR = 0.1         # Lagrangian multiplier learning rate
SAFETY_BUDGET = 0.01     # Target maximum Tier 1 violation rate (1%)
GRAD_CLIP_NORM = 10.0   # max gradient norm for QR-DQN
