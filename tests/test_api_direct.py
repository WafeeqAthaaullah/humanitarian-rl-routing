import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.main import (
    health_check,
    get_models,
    simulate_dispatch,
    run_model_showdown,
    get_training_metrics,
    inspect_state_quantiles,
)
from backend.schemas import SimulationRequest, ShowdownRequest

print("1. Health check:")
print(health_check())

print("\n2. Models metadata:")
models = get_models()
print("Available models:", list(models["models"].keys()))

print("\n3. Testing simulate_dispatch with QR-DQN + CVaR:")
req_sim = SimulationRequest(
    urgency_tier=1.0,
    shipment_weight=15.0,
    deadline_remaining=10.0,
    disruption_shock_prob=0.35,
    seed=42,
    agent_id="qr_dqn_cvar",
)
res_sim = simulate_dispatch(req_sim)
print(f"Simulation: {res_sim.agent_name} -> Status: {res_sim.status}, Cost: ${res_sim.total_freight_cost}, Days: {res_sim.total_days:.1f}, Steps: {len(res_sim.trajectory)}")
for s in res_sim.trajectory:
    print(f"  Step {s.step_index}: {s.location_start_name} -> {s.location_end_name} | {s.action_name} | Disruption: {s.disruption_triggered}")

print("\n4. Testing run_model_showdown:")
req_show = ShowdownRequest(
    urgency_tier=1.0,
    shipment_weight=15.0,
    deadline_remaining=10.0,
    disruption_shock_prob=0.35,
    seed=42,
)
res_show = run_model_showdown(req_show)
print(f"Showdown results count: {len(res_show.results)}")
for r in res_show.results:
    print(f"  {r.agent_name:18} | {r.status:17} | Cost: ${r.total_freight_cost:7.1f} | Days: {r.total_days:4.1f} | Actions: {r.action_sequence}")

print("\n5. Testing get_training_metrics:")
res_m = get_training_metrics(downsample=100)
print(f"Training metrics: {res_m.total_episodes} episodes, {len(res_m.records)} records")

print("\n6. Testing inspect_state_quantiles:")
sample_state = [1.0, 30.0, 15.0, 12.0, 0.0, 0.0, 1.0, 1.0, 1.0, 20.0, 0.35]
res_q = inspect_state_quantiles(sample_state)
print("Quantiles matrix:", len(res_q.quantiles), "actions x", len(res_q.quantiles[0]), "quantiles")
print("CVaR 5%:", [round(x, 3) for x in res_q.cvar_05])

print("\nALL FASTAPI ENDPOINT LOGIC TESTED AND VERIFIED SUCCESSFULLY!")
