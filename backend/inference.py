import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import pandas as pd

# Add repo root to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import (
    BASE_TRANSIT_TIMES,
    DISRUPTION_LOGNORM_MEAN,
    DISRUPTION_LOGNORM_STD,
    MIN_TRANSIT_TIME,
    TRANSIT_STD,
    STATE_DIM,
    ACTION_DIM,
    NUM_QUANTILES,
    MAX_AIR_CAPACITY,
    MAX_OCEAN_CAPACITY,
    MAX_LAND_CAPACITY,
)
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv
from agents.heuristics import GreedyCheapestAgent, StaticThresholdAgent
from agents.random_agent import RandomMaskedAgent
from agents.dqn_agent import DQNAgent
from agents.ddqn_lagrangian_agent import DDQNLagrangianAgent
from agents.qr_dqn_mean_agent import QRDQNMeanAgent
from agents.qr_dqn_agent import QRDQNAgent
from backend.schemas import (
    QuantileAnalysis,
    TrajectoryStep,
    SimulationResponse,
    ModelShowdownResult,
    ShowdownResponse,
    TrainingMetricsResponse,
    TrainingMetricRecord,
)

ACTION_NAMES = [
    "AIR_DIRECT",
    "OCEAN_TO_HUB",
    "LAND_TO_HUB",
    "SWITCH_TO_AIR",
    "CONTINUE_NOMINAL",
]

ACTION_DISPLAY_NAMES = [
    "Air Direct (Dubai -> Site)",
    "Ocean to Hub (Dubai -> Regional Checkpoint)",
    "Land to Hub (Dubai -> Regional Checkpoint)",
    "Switch to Air (Regional Checkpoint -> Site Pivot)",
    "Continue Nominal (Regional Checkpoint -> Site)",
]

LOCATION_NAMES = {
    0.0: "Dubai Origin",
    1.0: "Regional Checkpoint",
    2.0: "Disaster Site",
}

AGENT_METADATA = {
    "qr_dqn_cvar": {
        "name": "QR-DQN + CVaR",
        "family": "RL (Distributional Risk-Averse)",
        "type": "qr_dqn",
        "color": "#10b981",  # emerald-500
        "description": "Quantile Regression DQN selecting actions by optimizing the worst 10% (CVaR) tail return.",
    },
    "qr_dqn_mean": {
        "name": "QR-DQN-Mean",
        "family": "RL (Distributional Risk-Neutral)",
        "type": "qr_dqn_mean",
        "color": "#06b6d4",  # cyan-500
        "description": "Quantile Regression DQN selecting actions by the mean expectation over all 50 quantiles.",
    },
    "ddqn": {
        "name": "DDQN (Lagrangian)",
        "family": "RL (Value-Based Constrained)",
        "type": "ddqn",
        "color": "#3b82f6",  # blue-500
        "description": "Double DQN with static Lagrangian safety penalty (-50.0) for deadline misses.",
    },
    "dqn": {
        "name": "DQN",
        "family": "RL (Standard Value-Based)",
        "type": "dqn",
        "color": "#a855f7",  # purple-500
        "description": "Standard Deep Q-Network optimizing scalar expected return without tail-risk sensitivity.",
    },
    "static_threshold": {
        "name": "Static Threshold",
        "family": "Heuristic Rule-Based",
        "type": "heuristic",
        "color": "#f59e0b",  # amber-500
        "description": "Fixed heuristic: Dispatches by Air only if Tier 1 and deadline < 14 days; else cheapest.",
    },
    "greedy": {
        "name": "Greedy Cheapest",
        "family": "Heuristic Cost-First",
        "type": "heuristic",
        "color": "#ef4444",  # red-500
        "description": "Always selects lowest upfront freight rate (Ocean -> Continue Nominal), ignoring disruption risk.",
    },
    "random": {
        "name": "Random Masked",
        "family": "Stochastic Baseline",
        "type": "random",
        "color": "#64748b",  # slate-500
        "description": "Selects uniformly at random among feasible actions defined by the action mask.",
    },
}


class InstrumentedDubaiHumanitarianEnv(DubaiHumanitarianEnv):
    """
    Subclass that records whether a maritime disruption shock was activated
    during transit time calculation, keeping RNG streams identical.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_disruption_triggered = False
        self.last_sampled_delay = 0.0

    def _sample_transit_time(self, route: str, p_shock: float) -> float:
        is_disrupted = bool(self.np_random.random() < p_shock)
        mu = BASE_TRANSIT_TIMES.get(route, 5.0)

        if is_disrupted and "ocean" in route:
            self.last_disruption_triggered = True
            sampled = mu + float(
                self.np_random.lognormal(
                    mean=DISRUPTION_LOGNORM_MEAN, sigma=DISRUPTION_LOGNORM_STD
                )
            )
            self.last_sampled_delay = sampled
            return sampled

        self.last_disruption_triggered = False
        sampled = max(
            MIN_TRANSIT_TIME,
            float(self.np_random.normal(loc=mu, scale=TRANSIT_STD)),
        )
        self.last_sampled_delay = sampled
        return sampled

    def custom_reset(
        self,
        seed: Optional[int] = None,
        urgency_tier: Optional[float] = None,
        shipment_weight: Optional[float] = None,
        deadline_remaining: Optional[float] = None,
        disruption_shock_prob: Optional[float] = None,
        volume: Optional[float] = None,
        dest_inventory: Optional[float] = None,
        c_air: Optional[float] = None,
        c_ocean: Optional[float] = None,
        c_land: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.reset(seed=seed)
        self.last_disruption_triggered = False
        self.last_sampled_delay = 0.0

        if urgency_tier is not None:
            self.state[0] = float(urgency_tier)
        if volume is not None:
            self.state[1] = float(volume)
        if shipment_weight is not None:
            self.state[2] = float(shipment_weight)
        if deadline_remaining is not None:
            self.state[3] = float(deadline_remaining)
        if c_air is not None:
            self.state[6] = float(c_air)
        if c_ocean is not None:
            self.state[7] = float(c_ocean)
        if c_land is not None:
            self.state[8] = float(c_land)
        if dest_inventory is not None:
            self.state[9] = float(dest_inventory)
        if disruption_shock_prob is not None:
            self.state[10] = float(disruption_shock_prob)

        info["action_mask"] = self.get_action_mask()
        return self.state.copy(), info


class ModelRegistry:
    """Singleton cache for loaded agents."""
    _instance = None

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def _load_all(self):
        print("[Inference] Loading all 7 models...")
        # 1. Greedy
        self.models["greedy"] = GreedyCheapestAgent(None)

        # 2. Static Threshold
        self.models["static_threshold"] = StaticThresholdAgent(None)

        # 3. Random
        self.models["random"] = RandomMaskedAgent()

        # 4. DQN
        dqn_path = os.path.join(REPO_ROOT, "results", "checkpoints", "dqn_model.pth")
        dqn_agent = DQNAgent()
        if os.path.exists(dqn_path):
            ckpt = torch.load(dqn_path, map_location=dqn_agent.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            dqn_agent.network.load_state_dict(state_dict)
            dqn_agent.network.eval()
        self.models["dqn"] = dqn_agent

        # 5. DDQN
        ddqn_path = os.path.join(REPO_ROOT, "results", "checkpoints", "ddqn_lagrangian_model.pth")
        ddqn_agent = DDQNLagrangianAgent()
        if os.path.exists(ddqn_path):
            ckpt = torch.load(ddqn_path, map_location=ddqn_agent.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            ddqn_agent.online_net.load_state_dict(state_dict)
            ddqn_agent.online_net.eval()
        self.models["ddqn"] = ddqn_agent

        # 6. QR-DQN Mean
        qr_mean_path = os.path.join(REPO_ROOT, "results", "checkpoints", "qr_dqn_mean_model.pth")
        qr_mean_agent = QRDQNMeanAgent()
        if os.path.exists(qr_mean_path):
            ckpt = torch.load(qr_mean_path, map_location=qr_mean_agent.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            qr_mean_agent.network.load_state_dict(state_dict)
            qr_mean_agent.network.eval()
        self.models["qr_dqn_mean"] = qr_mean_agent

        # 7. QR-DQN + CVaR
        best_qr_path = os.path.join(REPO_ROOT, "results", "checkpoints", "best_feasible_qr_dqn_model.pth")
        if not os.path.exists(best_qr_path):
            best_qr_path = os.path.join(REPO_ROOT, "results", "checkpoints", "qr_dqn_model.pth")
        qr_cvar_agent = QRDQNAgent()
        if os.path.exists(best_qr_path):
            ckpt = torch.load(best_qr_path, map_location=qr_cvar_agent.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            qr_cvar_agent.network.load_state_dict(state_dict)
            qr_cvar_agent.network.eval()
        self.models["qr_dqn_cvar"] = qr_cvar_agent
        print("[Inference] All 7 models successfully initialized in memory.")

    def get_agent(self, agent_id: str) -> Any:
        return self.models.get(agent_id, self.models["qr_dqn_cvar"])

    def extract_quantiles_analysis(self, state: np.ndarray) -> QuantileAnalysis:
        """Extract full 5x50 quantile matrix, CVaR, and mean for any state vector."""
        qr_agent: QRDQNAgent = self.models["qr_dqn_cvar"]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(qr_agent.device)
        state_norm = qr_agent._normalize(state_tensor)

        with torch.no_grad():
            quantiles = qr_agent.network(state_norm)  # [1, 5, 50]
            q_matrix = quantiles.squeeze(0).cpu().numpy().tolist()  # [5, 50]

            sorted_quantiles, _ = torch.sort(quantiles, dim=2)
            # 5% CVaR: bottom 3 quantiles (or 2)
            cutoff_05 = max(1, int(round(0.05 * NUM_QUANTILES)))  # 3 quantiles for 50
            cvar_05_vals = sorted_quantiles[:, :, :cutoff_05].mean(dim=2).squeeze(0).cpu().numpy().tolist()

            # 10% CVaR: bottom 5 quantiles
            cutoff_10 = max(1, int(round(0.10 * NUM_QUANTILES)))  # 5 quantiles for 50
            cvar_10_vals = sorted_quantiles[:, :, :cutoff_10].mean(dim=2).squeeze(0).cpu().numpy().tolist()

            # Mean Q-values
            q_mean_vals = quantiles.mean(dim=2).squeeze(0).cpu().numpy().tolist()

        return QuantileAnalysis(
            action_names=ACTION_NAMES,
            quantiles=q_matrix,
            cvar_05=cvar_05_vals,
            cvar_10=cvar_10_vals,
            q_mean=q_mean_vals,
            cutoff_index_05=cutoff_05,
            cutoff_index_10=cutoff_10,
        )


def select_agent_action(agent_id: str, agent: Any, state: np.ndarray, mask: np.ndarray) -> int:
    meta = AGENT_METADATA.get(agent_id, {})
    atype = meta.get("type", "qr_dqn")

    if atype == "qr_dqn":
        act = agent.act_cvar(state, epsilon=0.0, action_mask=mask)
    elif atype == "qr_dqn_mean":
        act = agent.act_mean(state, epsilon=0.0, action_mask=mask)
    elif atype in ("dqn", "ddqn"):
        act = agent.act(state, epsilon=0.0, action_mask=mask)
    elif atype == "random":
        act = agent.act(state, action_mask=mask)
    else:  # heuristic
        act = agent.act(state)

    if not mask[act]:
        valid = np.flatnonzero(mask)
        act = int(valid[0]) if len(valid) > 0 else 0
    return int(act)


def get_mode_description(action: int, m_curr: float) -> str:
    if action == 0:
        return "Air Cargo (Direct)"
    elif action == 1:
        return "Ocean Freight (Leg 1 to Hub)"
    elif action == 2:
        return "Overland Trucking (Leg 1 to Hub)"
    elif action == 3:
        return "Emergency Intermodal Air Pivot (Leg 2)"
    elif action == 4:
        return "Ocean Continued (Leg 2)" if m_curr == 2.0 else "Land Continued (Leg 2)"
    return "Unknown"


def run_single_simulation(
    agent_id: str,
    urgency_tier: float = 1.0,
    shipment_weight: float = 15.0,
    deadline_remaining: float = 12.0,
    disruption_shock_prob: float = 0.35,
    seed: int = 42,
    volume: float = 30.0,
    dest_inventory: float = 20.0,
    c_air: float = 1.0,
    c_ocean: float = 1.0,
    c_land: float = 1.0,
) -> SimulationResponse:
    registry = ModelRegistry.get_instance()
    agent = registry.get_agent(agent_id)
    meta = AGENT_METADATA.get(agent_id, AGENT_METADATA["qr_dqn_cvar"])

    env = InstrumentedDubaiHumanitarianEnv()
    state, info = env.custom_reset(
        seed=seed,
        urgency_tier=urgency_tier,
        shipment_weight=shipment_weight,
        deadline_remaining=deadline_remaining,
        disruption_shock_prob=disruption_shock_prob,
        volume=volume,
        dest_inventory=dest_inventory,
        c_air=c_air,
        c_ocean=c_ocean,
        c_land=c_land,
    )

    initial_state = state.copy().tolist()
    initial_deadline = float(state[3])
    trajectory: List[TrajectoryStep] = []

    cumulative_cost = 0.0
    days_elapsed = 0.0
    pivoted = False
    disruption_occurred = False
    step_idx = 0
    done = False

    while not done and step_idx < 10:
        mask = env.get_action_mask()
        loc_start = float(state[4])
        loc_start_name = LOCATION_NAMES.get(loc_start, "Unknown")
        m_curr_before = float(state[5])

        quantiles_analysis = registry.extract_quantiles_analysis(state)
        action = select_agent_action(agent_id, agent, state, mask)

        if action == 3:
            pivoted = True

        next_state, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        step_cost = float(step_info.get("freight_cost", 0.0))
        days_passed = float(step_info.get("days_passed", 0.0))
        cumulative_cost += step_cost
        days_elapsed += days_passed

        loc_end = float(next_state[4])
        loc_end_name = LOCATION_NAMES.get(loc_end, "Unknown")

        disruption_on_step = getattr(env, "last_disruption_triggered", False)
        if disruption_on_step:
            disruption_occurred = True

        action_name = ACTION_NAMES[action]
        mode_str = get_mode_description(action, m_curr_before)

        step_note = ""
        if disruption_on_step:
            step_note += f"⚠️ MARITIME BOTTLENECK DISRUPTION TRIGGERED! Route delay: {days_passed:.1f} days. "
        if action == 3:
            step_note += "⚡ EMERGENCY INTERMODAL PIVOT: Switched cargo from maritime to air transport at Regional Checkpoint! "
        elif action == 0:
            step_note += "🚀 Direct expedited air delivery selected from Dubai Origin. "

        trajectory.append(
            TrajectoryStep(
                step_index=step_idx + 1,
                location_start=loc_start,
                location_end=loc_end,
                location_start_name=loc_start_name,
                location_end_name=loc_end_name,
                action=action,
                action_name=action_name,
                transport_mode=mode_str,
                freight_cost=step_cost,
                cumulative_cost=cumulative_cost,
                days_passed=days_passed,
                days_elapsed=days_elapsed,
                deadline_remaining=float(next_state[3]),
                state_vector=next_state.copy().tolist(),
                action_mask=mask.tolist(),
                disruption_triggered=disruption_on_step,
                is_pivot=bool(action == 3),
                deadline_hit=bool(step_info.get("deadline_hit", False)),
                tier1_violation=bool(step_info.get("tier1_violation", False)),
                quantiles_analysis=quantiles_analysis,
                step_notes=step_note.strip(),
            )
        )

        state = next_state
        step_idx += 1

    final_slack = float(state[3])
    is_tier1 = bool(initial_state[0] == 1.0)
    violated = bool((is_tier1 and final_slack < 0) or not step_info.get("deadline_hit", True))
    status = "Deadline Violated" if violated else "Delivered"

    scenario_dict = {
        "urgency_tier": urgency_tier,
        "shipment_weight": shipment_weight,
        "deadline_remaining": deadline_remaining,
        "disruption_shock_prob": disruption_shock_prob,
        "seed": seed,
        "volume": volume,
        "dest_inventory": dest_inventory,
    }

    return SimulationResponse(
        agent_id=agent_id,
        agent_name=meta["name"],
        agent_family=meta["family"],
        seed=seed,
        status=status,
        total_freight_cost=cumulative_cost,
        total_days=days_elapsed,
        initial_deadline=initial_deadline,
        remaining_slack=final_slack,
        pivoted=pivoted,
        disruption_occurred=disruption_occurred,
        trajectory=trajectory,
        initial_state=initial_state,
        final_state=state.copy().tolist(),
        scenario=scenario_dict,
    )


def run_showdown(
    urgency_tier: float = 1.0,
    shipment_weight: float = 15.0,
    deadline_remaining: float = 12.0,
    disruption_shock_prob: float = 0.35,
    seed: int = 42,
    volume: float = 30.0,
    dest_inventory: float = 20.0,
) -> ShowdownResponse:
    registry = ModelRegistry.get_instance()
    results: List[ModelShowdownResult] = []

    model_ids = [
        "greedy",
        "static_threshold",
        "random",
        "dqn",
        "ddqn",
        "qr_dqn_mean",
        "qr_dqn_cvar",
    ]

    for m_id in model_ids:
        meta = AGENT_METADATA[m_id]
        agent = registry.get_agent(m_id)
        env = InstrumentedDubaiHumanitarianEnv()
        state, info = env.custom_reset(
            seed=seed,
            urgency_tier=urgency_tier,
            shipment_weight=shipment_weight,
            deadline_remaining=deadline_remaining,
            disruption_shock_prob=disruption_shock_prob,
            volume=volume,
            dest_inventory=dest_inventory,
        )

        initial_u = float(state[0])
        cumulative_cost = 0.0
        days_elapsed = 0.0
        pivoted = False
        disruption_triggered = False
        action_sequence: List[int] = []
        action_names_list: List[str] = []
        trajectory_steps: List[TrajectoryStep] = []
        done = False
        step_idx = 0

        while not done and step_idx < 10:
            mask = env.get_action_mask()
            loc_start = float(state[4])
            loc_start_name = LOCATION_NAMES.get(loc_start, "Unknown")
            m_curr_before = float(state[5])

            action = select_agent_action(m_id, agent, state, mask)
            if action == 3:
                pivoted = True
            action_sequence.append(action)
            action_names_list.append(ACTION_NAMES[action])

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            step_cost = float(step_info.get("freight_cost", 0.0))
            days_passed = float(step_info.get("days_passed", 0.0))
            cumulative_cost += step_cost
            days_elapsed += days_passed

            loc_end = float(next_state[4])
            loc_end_name = LOCATION_NAMES.get(loc_end, "Unknown")
            disruption_on_step = getattr(env, "last_disruption_triggered", False)
            if disruption_on_step:
                disruption_triggered = True

            mode_str = get_mode_description(action, m_curr_before)

            trajectory_steps.append(
                TrajectoryStep(
                    step_index=step_idx + 1,
                    location_start=loc_start,
                    location_end=loc_end,
                    location_start_name=loc_start_name,
                    location_end_name=loc_end_name,
                    action=action,
                    action_name=ACTION_NAMES[action],
                    transport_mode=mode_str,
                    freight_cost=step_cost,
                    cumulative_cost=cumulative_cost,
                    days_passed=days_passed,
                    days_elapsed=days_elapsed,
                    deadline_remaining=float(next_state[3]),
                    state_vector=next_state.copy().tolist(),
                    action_mask=mask.tolist(),
                    disruption_triggered=disruption_on_step,
                    is_pivot=bool(action == 3),
                    deadline_hit=bool(step_info.get("deadline_hit", False)),
                    tier1_violation=bool(step_info.get("tier1_violation", False)),
                    quantiles_analysis=None,
                    step_notes="",
                )
            )

            state = next_state
            step_idx += 1

        final_slack = float(state[3])
        is_tier1 = bool(initial_u == 1.0)
        violated = bool((is_tier1 and final_slack < 0) or not step_info.get("deadline_hit", True))
        status = "Deadline Violated" if violated else "Delivered"

        results.append(
            ModelShowdownResult(
                agent_id=m_id,
                agent_name=meta["name"],
                family=meta["family"],
                status=status,
                total_freight_cost=cumulative_cost,
                total_days=days_elapsed,
                deadline_remaining=final_slack,
                pivoted=pivoted,
                disruption_triggered=disruption_triggered,
                action_sequence=action_sequence,
                action_names=action_names_list,
                trajectory=trajectory_steps,
            )
        )

    scenario_dict = {
        "urgency_tier": urgency_tier,
        "shipment_weight": shipment_weight,
        "deadline_remaining": deadline_remaining,
        "disruption_shock_prob": disruption_shock_prob,
        "seed": seed,
        "volume": volume,
        "dest_inventory": dest_inventory,
    }

    return ShowdownResponse(
        seed=seed,
        scenario=scenario_dict,
        results=results,
    )


def load_training_metrics(downsample_factor: int = 25) -> TrainingMetricsResponse:
    """Parse results/logs/training_metrics.csv with rolling averages."""
    csv_path = os.path.join(REPO_ROOT, "results", "logs", "training_metrics.csv")
    if not os.path.exists(csv_path):
        return TrainingMetricsResponse(
            total_episodes=0,
            records=[],
            rolling_violation_rate=[],
            rolling_reward=[],
            rolling_freight_cost=[],
        )

    df = pd.read_csv(csv_path)
    total_episodes = len(df)

    # Compute 100-episode rolling metrics
    df["rolling_violation"] = df["violation"].rolling(100, min_periods=1).mean() * 100.0
    df["rolling_reward"] = df["reward"].rolling(100, min_periods=1).mean()
    df["rolling_cost"] = df["freight_cost"].rolling(100, min_periods=1).mean()

    # Downsample for frontend performance
    sub_df = df.iloc[::downsample_factor]

    records = [
        TrainingMetricRecord(
            episode=int(row["episode"]),
            reward=float(row["reward"]),
            loss=float(row["loss"]),
            epsilon=float(row["epsilon"]),
            lambda_penalty=float(row["lambda_penalty"]),
            violation=int(row["violation"]),
            tier1=int(row["tier1"]),
            freight_cost=float(row["freight_cost"]),
            deadline_slack=float(row["deadline_slack"]),
            invalid_action=int(row["invalid_action"]),
        )
        for _, row in sub_df.iterrows()
    ]

    return TrainingMetricsResponse(
        total_episodes=total_episodes,
        records=records,
        rolling_violation_rate=sub_df["rolling_violation"].tolist(),
        rolling_reward=sub_df["rolling_reward"].tolist(),
        rolling_freight_cost=sub_df["rolling_cost"].tolist(),
    )
