from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Configuration for single-agent interactive simulation."""
    urgency_tier: float = Field(default=1.0, description="Urgency tier: 1.0 = Critical (Tier 1), 2.0 = Standard (Tier 2)")
    shipment_weight: float = Field(default=15.0, ge=0.5, le=50.0, description="Cargo weight in metric tons")
    deadline_remaining: float = Field(default=12.0, ge=1.0, le=35.0, description="Initial deadline in days")
    disruption_shock_prob: float = Field(default=0.35, ge=0.0, le=0.5, description="Probability of maritime route disruption [0.0 - 0.5]")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducible stochastic delays")
    agent_id: str = Field(default="qr_dqn_cvar", description="Selected agent: greedy, static_threshold, random, dqn, ddqn, qr_dqn_mean, qr_dqn_cvar")
    volume: Optional[float] = Field(default=30.0, ge=1.0, le=100.0, description="Shipment volume in m^3")
    dest_inventory: Optional[float] = Field(default=20.0, ge=0.0, le=100.0, description="Destination buffer inventory in days")
    c_air: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Air capacity ratio [0.0 - 1.0]")
    c_ocean: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Ocean capacity ratio [0.0 - 1.0]")
    c_land: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Land capacity ratio [0.0 - 1.0]")


class QuantileAnalysis(BaseModel):
    """5x50 quantile matrix and risk metrics."""
    action_names: List[str]
    quantiles: List[List[float]] = Field(description="5 actions x 50 quantiles matrix Z_1...Z_50")
    cvar_05: List[float] = Field(description="5% CVaR for each of the 5 actions (worst ~3 quantiles)")
    cvar_10: List[float] = Field(description="10% CVaR for each of the 5 actions (worst 5 quantiles)")
    q_mean: List[float] = Field(description="Mean expected value for each of the 5 actions")
    cutoff_index_05: int
    cutoff_index_10: int


class TrajectoryStep(BaseModel):
    """Detailed record of an individual step in the multi-echelon journey."""
    step_index: int
    location_start: float
    location_end: float
    location_start_name: str
    location_end_name: str
    action: int
    action_name: str
    transport_mode: str
    freight_cost: float
    cumulative_cost: float
    days_passed: float
    days_elapsed: float
    deadline_remaining: float
    state_vector: List[float]
    action_mask: List[bool]
    disruption_triggered: bool
    is_pivot: bool
    deadline_hit: bool
    tier1_violation: bool
    quantiles_analysis: Optional[QuantileAnalysis] = None
    step_notes: str = ""


class SimulationResponse(BaseModel):
    """Full execution trajectory of an interactive dispatch simulation."""
    agent_id: str
    agent_name: str
    agent_family: str
    seed: int
    status: str  # "Delivered" or "Deadline Violated"
    total_freight_cost: float
    total_days: float
    initial_deadline: float
    remaining_slack: float
    pivoted: bool
    disruption_occurred: bool
    trajectory: List[TrajectoryStep]
    initial_state: List[float]
    final_state: List[float]
    scenario: Dict[str, Any]


class ShowdownRequest(BaseModel):
    """Configuration to run all 7 models simultaneously on the same scenario."""
    urgency_tier: float = Field(default=1.0, description="1.0 = Critical (Tier 1), 2.0 = Standard (Tier 2)")
    shipment_weight: float = Field(default=15.0, ge=0.5, le=50.0)
    deadline_remaining: float = Field(default=12.0, ge=1.0, le=35.0)
    disruption_shock_prob: float = Field(default=0.35, ge=0.0, le=0.5)
    seed: Optional[int] = Field(default=42)
    volume: Optional[float] = Field(default=30.0)
    dest_inventory: Optional[float] = Field(default=20.0)


class ModelShowdownResult(BaseModel):
    """Individual model outcome in the multi-agent showdown."""
    agent_id: str
    agent_name: str
    family: str  # "RL (Distributional)", "RL (Value-Based)", "Heuristic", "Baseline"
    status: str  # "Delivered" or "Deadline Violated"
    total_freight_cost: float
    total_days: float
    deadline_remaining: float
    pivoted: bool
    disruption_triggered: bool
    action_sequence: List[int]
    action_names: List[str]
    trajectory: List[TrajectoryStep]


class ShowdownResponse(BaseModel):
    """Aggregated side-by-side comparison across all 7 agents."""
    seed: int
    scenario: Dict[str, Any]
    results: List[ModelShowdownResult]


class TrainingMetricRecord(BaseModel):
    episode: int
    reward: float
    loss: float
    epsilon: float
    lambda_penalty: float
    violation: int
    tier1: int
    freight_cost: float
    deadline_slack: float
    invalid_action: int


class TrainingMetricsResponse(BaseModel):
    total_episodes: int
    records: List[TrainingMetricRecord]
    rolling_violation_rate: List[float]
    rolling_reward: List[float]
    rolling_freight_cost: List[float]
