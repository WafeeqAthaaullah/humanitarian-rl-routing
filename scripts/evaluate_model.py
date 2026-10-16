"""Repeated-seed benchmark evaluation for the humanitarian routing agents."""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.heuristics import (
    AlwaysAirAgent,
    AlwaysOceanAgent,
    GreedyCheapestAgent,
    StaticThresholdAgent,
)
from agents.qr_dqn_agent import QRDQNAgent
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv


def evaluate_agent(agent, env, agent_type="heuristic", episodes=500, seed=0):
    total_cost = 0.0
    violations = 0
    tier1_count = 0
    invalid_actions = 0
    total_delay = 0.0
    deadline_slack = []

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        is_tier1 = bool(state[0] == 1.0)
        tier1_count += int(is_tier1)
        while True:
            if agent_type == "rl":
                action = agent.act_cvar(
                    state, epsilon=0.0, action_mask=info["action_mask"]
                )
            else:
                action = agent.act(state)
                if not info["action_mask"][action]:
                    action = int(np.flatnonzero(info["action_mask"])[0])

            state, _, terminated, truncated, info = env.step(action)
            total_cost += float(info.get("freight_cost", 0.0))
            total_delay += float(info.get("days_passed", 0.0))
            deadline_slack.append(float(info.get("deadline_slack", 0.0)))
            if "reason" in info:
                invalid_actions += 1
            if is_tier1 and info.get("tier1_violation", False):
                violations += 1
            if terminated or truncated:
                break

    return {
        "violation_rate": 100.0 * violations / max(1, tier1_count),
        "average_cost": total_cost / max(1, episodes),
        "average_delay": total_delay / max(1, episodes),
        "average_deadline_slack": float(np.mean(deadline_slack)),
        "invalid_action_rate": 100.0 * invalid_actions / max(1, episodes),
        "tier1_count": tier1_count,
    }


def load_rl_agent(checkpoint_path):
    agent = QRDQNAgent()
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    agent.network.load_state_dict(state_dict)
    agent.network.eval()
    return agent


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    margin = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return float(np.mean(values)), margin


def run_benchmark(checkpoint_path, episodes, seeds):
    policies = {
        "Always-Air": (AlwaysAirAgent, "heuristic"),
        "Always-Ocean": (AlwaysOceanAgent, "heuristic"),
        "Greedy-Cheapest": (GreedyCheapestAgent, "heuristic"),
        "Static Threshold": (lambda env: StaticThresholdAgent(env, urgency_threshold=12.0), "heuristic"),
        "QR-DQN (Ours)": (lambda env: load_rl_agent(checkpoint_path), "rl"),
    }
    aggregate = {}
    for name, (factory, agent_type) in policies.items():
        runs = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            env = DubaiHumanitarianEnv()
            runs.append(evaluate_agent(factory(env), env, agent_type, episodes, seed))
        aggregate[name] = runs
        violation, violation_ci = confidence_interval([r["violation_rate"] for r in runs])
        cost, cost_ci = confidence_interval([r["average_cost"] for r in runs])
        print(
            f"{name:16s} | Violation: {violation:6.2f}% +/- {violation_ci:.2f} "
            f"| Cost: ${cost:,.2f} +/- ${cost_ci:,.2f}"
        )

    labels = list(aggregate)
    violations = [np.mean([r["violation_rate"] for r in aggregate[k]]) for k in labels]
    costs = [np.mean([r["average_cost"] for r in aggregate[k]]) for k in labels]
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, violations, width=0.4, color="tab:red")
    ax1.set_ylabel("Tier 1 Deadline Violation Rate (%)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, costs, width=0.4, color="tab:blue", alpha=0.7)
    ax2.set_ylabel("Average Freight Cost ($)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title("Repeated-Seed Agent Performance")
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/performance_comparison.png", dpi=200)
    plt.close()
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/qr_dqn_model.pth",
    )
    args = parser.parse_args()
    seed_values = list(range(args.seeds))
    print(f"Evaluating {len(seed_values)} seeds x {args.episodes} episodes...")
    run_benchmark(args.checkpoint, args.episodes, seed_values)
