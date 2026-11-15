import type {
  SimulationRequest,
  SimulationResponse,
  ShowdownRequest,
  ShowdownResponse,
  TrainingMetricsResponse,
  QuantileAnalysis,
  AgentMeta,
} from "./types";

const API_BASE = "/api";

export async function fetchSimulation(req: SimulationRequest): Promise<SimulationResponse> {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Simulation failed: ${res.statusText} - ${errorText}`);
  }
  return res.json();
}

export async function fetchShowdown(req: ShowdownRequest): Promise<ShowdownResponse> {
  const res = await fetch(`${API_BASE}/showdown`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Showdown failed: ${res.statusText} - ${errorText}`);
  }
  return res.json();
}

export async function fetchTrainingMetrics(downsample: number = 25): Promise<TrainingMetricsResponse> {
  const res = await fetch(`${API_BASE}/training-metrics?downsample=${downsample}`);
  if (!res.ok) {
    throw new Error(`Failed to load training metrics: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchModels(): Promise<{
  models: Record<string, AgentMeta>;
  action_names: string[];
  action_display_names: string[];
}> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    throw new Error(`Failed to fetch models: ${res.statusText}`);
  }
  return res.json();
}

export async function inspectQuantiles(state: number[]): Promise<QuantileAnalysis> {
  const res = await fetch(`${API_BASE}/quantile-inspection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
  if (!res.ok) {
    throw new Error(`Quantile inspection failed: ${res.statusText}`);
  }
  return res.json();
}
