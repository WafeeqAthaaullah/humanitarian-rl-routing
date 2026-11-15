import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.inference import run_single_simulation, run_showdown, load_training_metrics, ModelRegistry

print("1. Testing ModelRegistry...")
registry = ModelRegistry.get_instance()
print("Models loaded:", list(registry.models.keys()))

print("\n2. Testing run_single_simulation (QR-DQN + CVaR)...")
res_sim = run_single_simulation(
    agent_id="qr_dqn_cvar",
    urgency_tier=1.0,
    shipment_weight=15.0,
    deadline_remaining=10.0,
    disruption_shock_prob=0.35,
    seed=42,
)
print("Simulation result:", res_sim.status, "Cost:", res_sim.total_freight_cost, "Days:", res_sim.total_days)
print("Trajectory steps:", len(res_sim.trajectory))
for step in res_sim.trajectory:
    print(f" Step {step.step_index}: {step.location_start_name} -> {step.location_end_name} | Action: {step.action_name} | Disruption: {step.disruption_triggered}")
    if step.quantiles_analysis:
        print(f"   CVaR 5%: {step.quantiles_analysis.cvar_05}")
        print(f"   Mean Q: {step.quantiles_analysis.q_mean}")

print("\n3. Testing run_showdown (all 7 models)...")
res_show = run_showdown(
    urgency_tier=1.0,
    shipment_weight=15.0,
    deadline_remaining=10.0,
    disruption_shock_prob=0.35,
    seed=42,
)
print("Showdown results count:", len(res_show.results))
for r in res_show.results:
    print(f"  {r.agent_name:18} | {r.status:17} | Cost: ${r.total_freight_cost:7.1f} | Days: {r.total_days:4.1f} | Actions: {r.action_sequence}")

print("\n4. Testing load_training_metrics...")
metrics = load_training_metrics(downsample_factor=50)
print(f"Total training episodes: {metrics.total_episodes}, sampled records: {len(metrics.records)}")

print("\nALL BACKEND INFERENCE CHECKS PASSED SUCCESSFULLY!")
