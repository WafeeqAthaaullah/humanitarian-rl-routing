export interface SimulationRequest {
  urgency_tier: number; // 1.0 = Critical (Tier 1), 2.0 = Standard (Tier 2)
  shipment_weight: number; // 0.5 - 50.0 tons
  deadline_remaining: number; // 5.0 - 30.0 days
  disruption_shock_prob: number; // 0.0 - 0.5
  seed?: number;
  agent_id: string;
  volume?: number;
  dest_inventory?: number;
  c_air?: number;
  c_ocean?: number;
  c_land?: number;
}

export interface QuantileAnalysis {
  action_names: string[];
  quantiles: number[][]; // 5 x 50
  cvar_05: number[];
  cvar_10: number[];
  q_mean: number[];
  cutoff_index_05: number;
  cutoff_index_10: number;
}

export interface TrajectoryStep {
  step_index: number;
  location_start: number;
  location_end: number;
  location_start_name: string;
  location_end_name: string;
  action: number;
  action_name: string;
  transport_mode: string;
  freight_cost: number;
  cumulative_cost: number;
  days_passed: number;
  days_elapsed: number;
  deadline_remaining: number;
  state_vector: number[];
  action_mask: boolean[];
  disruption_triggered: boolean;
  is_pivot: boolean;
  deadline_hit: boolean;
  tier1_violation: boolean;
  quantiles_analysis?: QuantileAnalysis | null;
  step_notes: string;
}

export interface SimulationResponse {
  agent_id: string;
  agent_name: string;
  agent_family: string;
  seed: number;
  status: "Delivered" | "Deadline Violated" | string;
  total_freight_cost: number;
  total_days: number;
  initial_deadline: number;
  remaining_slack: number;
  pivoted: boolean;
  disruption_occurred: boolean;
  trajectory: TrajectoryStep[];
  initial_state: number[];
  final_state: number[];
  scenario: {
    urgency_tier: number;
    shipment_weight: number;
    deadline_remaining: number;
    disruption_shock_prob: number;
    seed: number;
    volume: number;
    dest_inventory: number;
  };
}

export interface ShowdownRequest {
  urgency_tier: number;
  shipment_weight: number;
  deadline_remaining: number;
  disruption_shock_prob: number;
  seed?: number;
  volume?: number;
  dest_inventory?: number;
}

export interface ModelShowdownResult {
  agent_id: string;
  agent_name: string;
  family: string;
  status: "Delivered" | "Deadline Violated" | string;
  total_freight_cost: number;
  total_days: number;
  deadline_remaining: number;
  pivoted: boolean;
  disruption_triggered: boolean;
  action_sequence: number[];
  action_names: string[];
  trajectory: TrajectoryStep[];
}

export interface ShowdownResponse {
  seed: number;
  scenario: {
    urgency_tier: number;
    shipment_weight: number;
    deadline_remaining: number;
    disruption_shock_prob: number;
    seed: number;
  };
  results: ModelShowdownResult[];
}

export interface TrainingMetricRecord {
  episode: number;
  reward: number;
  loss: number;
  epsilon: number;
  lambda_penalty: number;
  violation: number;
  tier1: number;
  freight_cost: number;
  deadline_slack: number;
  invalid_action: number;
}

export interface TrainingMetricsResponse {
  total_episodes: number;
  records: TrainingMetricRecord[];
  rolling_violation_rate: number[];
  rolling_reward: number[];
  rolling_freight_cost: number[];
}

export interface AgentMeta {
  name: string;
  family: string;
  type: string;
  color: string;
  description: string;
}
