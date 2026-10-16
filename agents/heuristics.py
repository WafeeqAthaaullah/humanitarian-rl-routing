import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from config import MAX_AIR_CAPACITY, MAX_OCEAN_CAPACITY, MAX_LAND_CAPACITY

class GreedyCheapestAgent:
    """
    Always picks the cheapest mode: Ocean to Hub, then Continue.
    If capacity is full, it falls back to the next cheapest.
    """
    def __init__(self, env):
        self.env = env

    def act(self, state):
        loc = state[4]
        c_air, c_ocean, c_land = state[6], state[7], state[8]
        w = state[2]

        if loc == 0.0:  # At Origin
            if c_ocean >= (w / MAX_OCEAN_CAPACITY):
                return 1  # Ocean (Cheapest)
            elif c_land >= (w / MAX_LAND_CAPACITY):
                return 2  # Land (Second Cheapest)
            return 0      # Air (Fallback)
            
        elif loc == 1.0:  # At Regional Checkpoint
            return 4      # Continue Nominal (Cheapest)

class StaticThresholdAgent:
    """
    Rule-based: Pick Air if it's Tier 1 (Critical) AND deadline is tight.
    Otherwise, default to cheapest.
    """
    def __init__(self, env, urgency_threshold=14.0):
        self.env = env
        self.tau = urgency_threshold

    def act(self, state):
        u = state[0]
        d_rem = state[3]
        loc = state[4]
        w = state[2]
        c_air = state[6]

        is_critical_and_urgent = (u == 1.0) and (d_rem < self.tau)
        can_use_air = c_air >= (w / MAX_AIR_CAPACITY)

        if loc == 0.0:
            if is_critical_and_urgent and can_use_air:
                return 0  # Air Direct
            return 1      # Default to Ocean

        elif loc == 1.0:
            if is_critical_and_urgent and can_use_air:
                return 3  # Switch to Air
            return 4      # Continue


class AlwaysAirAgent:
    """Safety-first baseline that uses air whenever it is feasible."""

    def __init__(self, env):
        self.env = env

    def act(self, state):
        return 0 if state[4] == 0.0 else 3


class AlwaysOceanAgent:
    """Cost-first baseline that uses ocean and then continues nominally."""

    def __init__(self, env):
        self.env = env

    def act(self, state):
        return 1 if state[4] == 0.0 else 4