import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    MAX_AIR_CAPACITY, MAX_OCEAN_CAPACITY, MAX_LAND_CAPACITY,
    FREIGHT_RATES, BASE_TRANSIT_TIMES,
    MODE_SWITCH_COST, MODE_SWITCH_DAYS,
    INVALID_ACTION_PENALTY, SAFETY_PENALTY, COST_NORMALIZATION,
    TRANSIT_STD, DISRUPTION_LOGNORM_MEAN, DISRUPTION_LOGNORM_STD,
    MIN_TRANSIT_TIME, OBS_LOW, OBS_HIGH,
)

class DubaiHumanitarianEnv(gym.Env):
    """
    Custom Environment for Humanitarian Transport-Mode Selection
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, config=None):
        super().__init__()
        
        # Action Space: 5 discrete actions as defined in the CMDP
        # 0: AIR_DIRECT, 1: OCEAN_TO_HUB, 2: LAND_TO_HUB, 3: SWITCH_TO_AIR, 4: CONTINUE_NOMINAL
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: 11-dimensional continuous/discrete hybrid vector
        # [u, v, w, d_rem, l, m_curr, c_air, c_ocean, c_land, I_dest, S_disrupt]
        # We use a Box space (float32) for standard neural network compatibility
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Randomize initial state so the agent learns across diverse scenarios
        u = float(self.np_random.choice([1, 2]))       # Urgency tier (1=Critical, 2=Standard)
        v = self.np_random.uniform(1.0, 100.0)          # Volume
        w = self.np_random.uniform(0.5, 50.0)           # Weight (tons)
        d_rem = self.np_random.uniform(5.0, 30.0)       # Deadline remaining (days)
        s_disrupt = self.np_random.uniform(0.0, 0.5)    # Disruption/shock probability
        i_dest = self.np_random.uniform(0.0, 100.0)     # Destination inventory buffer (days)

        self.state = np.array([
            u, v, w, d_rem,
            0.0,    # Location (0 = Origin)
            0.0,    # Current mode (0 = None)
            1.0,    # Air capacity (100%)
            1.0,    # Ocean capacity (100%)
            1.0,    # Land capacity (100%)
            i_dest,
            s_disrupt,
        ], dtype=np.float32)
        
        info = {"action_mask": self.get_action_mask()}
        return self.state, info

    def step(self, action):
        # Unpack current state for readability
        u, v, w, d_rem, loc, m_curr, c_air, c_ocean, c_land, i_dest, s_disrupt = self.state
        
        # 1. Action Masking & Feasibility Check
        if not self._is_action_valid(action):
            # Severe penalty for picking an invalid/impossible action, end episode
            reward = INVALID_ACTION_PENALTY
            return self.state, reward, True, False, {
                "reason": "invalid_action",
                "freight_cost": 0.0,
                "days_passed": 0.0,
                "deadline_hit": False,
                "tier1_violation": bool(u == 1.0),
                "action_mask": self.get_action_mask(),
            }

        # 2. Transition Dynamics Setup
        days_passed = 0
        step_cost = 0.0
        terminated = False
        
        # Process action effects
        if action == 0:  # AIR_DIRECT
            if c_air < (w / MAX_AIR_CAPACITY):
                return self.state, INVALID_ACTION_PENALTY, True, False, {
                    "reason": "capacity_exceeded",
                    "freight_cost": 0.0,
                    "days_passed": 0.0,
                    "deadline_hit": False,
                    "tier1_violation": bool(u == 1.0),
                    "action_mask": self.get_action_mask(),
                }
            c_air -= (w / MAX_AIR_CAPACITY)
            days_passed = self._sample_transit_time("air", s_disrupt)
            step_cost = self._get_freight_cost("air", w)
            loc = 2.0  # Arrived at destination
            terminated = True
            
        elif action == 1:  # OCEAN_TO_HUB
            if c_ocean < (w / MAX_OCEAN_CAPACITY):
                return self.state, INVALID_ACTION_PENALTY, True, False, {
                    "reason": "capacity_exceeded",
                    "freight_cost": 0.0,
                    "days_passed": 0.0,
                    "deadline_hit": False,
                    "tier1_violation": bool(u == 1.0),
                    "action_mask": self.get_action_mask(),
                }
            c_ocean -= (w / MAX_OCEAN_CAPACITY)
            days_passed = self._sample_transit_time("ocean_leg1", s_disrupt)
            step_cost = self._get_freight_cost("ocean", w)
            loc = 1.0  # Arrived at Regional Checkpoint
            m_curr = 2.0 # Mode is now Ocean
            
        elif action == 2:  # LAND_TO_HUB
            if c_land < (w / MAX_LAND_CAPACITY):
                return self.state, INVALID_ACTION_PENALTY, True, False, {
                    "reason": "capacity_exceeded",
                    "freight_cost": 0.0,
                    "days_passed": 0.0,
                    "deadline_hit": False,
                    "tier1_violation": bool(u == 1.0),
                    "action_mask": self.get_action_mask(),
                }
            c_land -= (w / MAX_LAND_CAPACITY)
            days_passed = self._sample_transit_time("land_leg1", s_disrupt)
            step_cost = self._get_freight_cost("land", w)
            loc = 1.0  # Arrived at Regional Checkpoint
            m_curr = 3.0 # Mode is now Land

        elif action == 3:  # SWITCH_TO_AIR (At Regional Checkpoint)
            if c_air < (w / MAX_AIR_CAPACITY):
                return self.state, INVALID_ACTION_PENALTY, True, False, {
                    "reason": "capacity_exceeded",
                    "freight_cost": 0.0,
                    "days_passed": 0.0,
                    "deadline_hit": False,
                    "tier1_violation": bool(u == 1.0),
                    "action_mask": self.get_action_mask(),
                }
            c_air -= (w / MAX_AIR_CAPACITY)
            days_passed = self._sample_transit_time("air", s_disrupt) + MODE_SWITCH_DAYS
            step_cost = self._get_freight_cost("air", w) + MODE_SWITCH_COST
            loc = 2.0
            terminated = True

        elif action == 4:  # CONTINUE_NOMINAL (At Regional Checkpoint)
            # Continues on whatever mode it arrived on
            mode_str = "ocean_leg2" if m_curr == 2.0 else "land_leg2"
            days_passed = self._sample_transit_time(mode_str, s_disrupt)
            step_cost = self._get_freight_cost("continue", w)
            loc = 2.0
            terminated = True

        # 3. Apply Time Effects
        d_rem -= days_passed
        # Inventory decays by an assumed fixed burn rate (e.g., 1 day of buffer consumed per day)
        i_dest = max(0.0, i_dest - days_passed)

        # 4. Reward Calculation (CMDP formulation)
        # Cost is scaled down so neural network gradients don't explode
        normalized_cost = step_cost / COST_NORMALIZATION 
        
        # Lagrangian safety penalty: Heavy penalty if Tier 1 (Critical) misses deadline
        safety_penalty = 0.0
        if terminated and loc == 2.0: 
            if u == 1.0 and d_rem < 0:
                # In the training loop, you will multiply this by your dynamic Lambda
                safety_penalty = SAFETY_PENALTY

        reward = - (normalized_cost + safety_penalty)

        # 5. Re-pack state
        self.state = np.array([
            u, v, w, d_rem, loc, m_curr, 
            max(0.0, c_air), max(0.0, c_ocean), max(0.0, c_land), 
            i_dest, s_disrupt
        ], dtype=np.float32)
        
        info = {
            "days_passed": days_passed,
            "deadline_hit": d_rem >= 0,
            "freight_cost": step_cost,
            "safety_penalty": safety_penalty,
            "tier1_violation": bool(u == 1.0 and d_rem < 0),
            "deadline_slack": d_rem,
            "action_mask": self.get_action_mask(),
        }
        
        return self.state, reward, terminated, False, info

    def _sample_transit_time(self, route, p_shock):
        """Mixture model for delays: Normal vs LogNormal disruption.
        
        Uses self.np_random (seeded in reset()) for full reproducibility.
        """
        is_disrupted = self.np_random.random() < p_shock
        
        mu = BASE_TRANSIT_TIMES.get(route, 5.0)
        
        if is_disrupted and "ocean" in route:
            # Fat-tailed delay for maritime bottlenecks
            return mu + self.np_random.lognormal(
                mean=DISRUPTION_LOGNORM_MEAN, sigma=DISRUPTION_LOGNORM_STD
            )
        
        # Normal variance
        return max(MIN_TRANSIT_TIME, self.np_random.normal(loc=mu, scale=TRANSIT_STD))

    def _get_freight_cost(self, mode, weight):
        """Rough simulated costs ($ per ton)."""
        return FREIGHT_RATES.get(mode, 500.0) * weight

    def _is_action_valid(self, action):
        action = int(action)
        if not 0 <= action < self.action_space.n:
            return False
        return bool(self.get_action_mask()[action])

    def get_action_mask(self):
        """Return legal and capacity-feasible actions for the current state."""
        if not hasattr(self, "state"):
            return np.ones(self.action_space.n, dtype=bool)
        location = self.state[4]
        weight = self.state[2]
        capacities = self.state[6:9]
        required = np.array(
            [weight / MAX_AIR_CAPACITY, weight / MAX_OCEAN_CAPACITY,
             weight / MAX_LAND_CAPACITY, weight / MAX_AIR_CAPACITY, 0.0]
        )
        mask = capacities[[0, 1, 2, 0, 0]] >= required
        if location == 0.0:
            mask[[3, 4]] = False
        elif location == 1.0:
            mask[[0, 1, 2]] = False
        else:
            mask[:] = False
        return mask.astype(bool)

if __name__ == "__main__":
    env = DubaiHumanitarianEnv()
    obs, info = env.reset()
    print("Initial State:", obs)
    print("Initial Info:", info)
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nAction: {action}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print(f"New State: {obs}")
        
        if terminated or truncated:
            print(">>> EPISODE TERMINATED <<<")
            break