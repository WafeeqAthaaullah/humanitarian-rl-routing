"""Random masked baseline agent for DubaiHumanitarianEnv."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np


class RandomMaskedAgent:
    """
    Random baseline agent that selects uniformly among legal actions
    defined by the environment's action mask for the current state.
    """

    def __init__(self, env=None):
        self.env = env

    def act(self, state, action_mask=None):
        """
        Select an action uniformly at random from valid actions.

        Parameters
        ----------
        state : np.ndarray
            Current environment state vector.
        action_mask : np.ndarray or None, optional
            Boolean mask of valid actions. If not provided and self.env is set,
            the mask is retrieved directly from self.env.get_action_mask().

        Returns
        -------
        int
            Selected valid action index.
        """
        if action_mask is None and self.env is not None and hasattr(self.env, "get_action_mask"):
            action_mask = self.env.get_action_mask()

        if action_mask is not None:
            action_mask = np.asarray(action_mask, dtype=bool)
            valid_actions = np.flatnonzero(action_mask)
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))

        # Fallback: if env has action_space or default to discrete 5 actions
        if self.env is not None and hasattr(self.env, "action_space"):
            return int(self.env.action_space.sample())
        return 0

    def select_action(self, state, action_mask=None):
        """Alias for act() for interface consistency."""
        return self.act(state, action_mask=action_mask)
