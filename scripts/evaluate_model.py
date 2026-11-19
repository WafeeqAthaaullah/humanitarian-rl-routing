"""Repeated-seed benchmark evaluation for all humanitarian routing agents."""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.ddqn_lagrangian_agent import DDQNLagrangianAgent
from agents.dqn_agent import DQNAgent
from agents.heuristics import GreedyCheapestAgent, StaticThresholdAgent
from agents.qr_dqn_agent import QRDQNAgent
from agents.qr_dqn_mean_agent import QRDQNMeanAgent
from agents.random_agent import RandomMaskedAgent
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv


def evaluate_agent(agent, env, agent_type="heuristic", episodes=500, seed=0, regime="nominal"):
    """
    Evaluate an agent across a given number of seeded Monte Carlo episodes.

    Parameters
    ----------
    agent : object
        Agent instance implementing act(), act_cvar(), act_mean(), or select_action().
    env : DubaiHumanitarianEnv
        Environment instance.
    agent_type : str
        Agent family: 'heuristic', 'random', 'dqn', 'ddqn_lagrangian', 'qr_dqn_mean', or 'qr_dqn'.
    episodes : int
        Number of test episodes.
    seed : int
        Starting random seed for reproducibility.
    regime : str
        Evaluation regime: 'nominal' (standard environment) or 'stress' (high disruption shock & tight deadline).

    Returns
    -------
    dict
        Evaluation metrics including Tier-1 violation rate and average cost.
    """
    total_cost = 0.0
    violations = 0
    tier1_count = 0
    invalid_actions = 0
    total_delay = 0.0
    deadline_slack = []

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        
        # High-Disruption Stress Regime: Tests tail-risk resilience under adverse bottleneck conditions
        if regime == "stress":
            rng = np.random.RandomState(seed + episode)
            state[3] = float(rng.uniform(6.0, 14.0))   # Tight deadline: 6 to 14 days
            state[10] = float(rng.uniform(0.35, 0.50)) # Severe disruption risk: 35% to 50% shock prob
            env.state = state
            info["action_mask"] = env.get_action_mask()

        is_tier1 = bool(state[0] == 1.0)
        tier1_count += int(is_tier1)
        while True:
            mask = info.get("action_mask", env.get_action_mask())

            if agent_type == "qr_dqn":
                action = agent.act_cvar(state, epsilon=0.0, action_mask=mask)
            elif agent_type == "qr_dqn_mean":
                action = agent.act_mean(state, epsilon=0.0, action_mask=mask)
            elif agent_type in ("dqn", "ddqn_lagrangian"):
                action = agent.act(state, epsilon=0.0, action_mask=mask)
            elif agent_type == "random":
                action = agent.act(state, action_mask=mask)
            else:  # heuristic
                action = agent.act(state)

            # Safety fallback if an action violates the mask
            if not mask[action]:
                action = int(np.flatnonzero(mask)[0])

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


def load_dqn_agent(checkpoint_path="results/checkpoints/dqn_model.pth"):
    agent = DQNAgent()
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    agent.network.load_state_dict(state_dict)
    agent.network.eval()
    return agent


def load_ddqn_lagrangian_agent(checkpoint_path="results/checkpoints/ddqn_lagrangian_model.pth"):
    agent = DDQNLagrangianAgent()
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    agent.online_net.load_state_dict(state_dict)
    agent.online_net.eval()
    return agent


def load_qr_dqn_mean_agent(checkpoint_path="results/checkpoints/qr_dqn_mean_model.pth"):
    agent = QRDQNMeanAgent()
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    agent.network.load_state_dict(state_dict)
    agent.network.eval()
    return agent


def load_qr_dqn_agent(checkpoint_path="results/checkpoints/qr_dqn_model.pth"):
    agent = QRDQNAgent()
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
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


def plot_metrics(labels, metrics, output_dir="results/plots", title_suffix="", filename_suffix=""):
    """
    Generate and save two separate, publication-quality bar charts:
    1. violation_rate_comparison{filename_suffix}.png: Tier 1 Deadline Violation Rate (%)
    2. freight_cost_comparison{filename_suffix}.png: Average Freight Cost ($)
    """
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(len(labels))
    bar_width = 0.55

    # Color palette tailored for scientific publication
    palette = [
        "#7f8c8d",  # Random (Neutral slate gray)
        "#e74c3c",  # Greedy Cheapest (Crimson red / High risk)
        "#e67e22",  # Static Threshold (Amber orange)
        "#9b59b6",  # DQN (Purple)
        "#2980b9",  # DDQN-Lagrangian (Deep blue)
        "#1abc9c",  # QR-DQN Mean (Teal)
        "#27ae60",  # QR-DQN CVaR (Emerald green / Robust)
    ]
    colors = palette[:len(labels)]

    # -------------------------------------------------------------
    # Plot 1: Tier 1 Deadline Violation Rate (%)
    # -------------------------------------------------------------
    v_means = [metrics[k]["violation_mean"] for k in labels]
    v_cis = [metrics[k]["violation_ci"] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x,
        v_means,
        yerr=v_cis if any(ci > 0 for ci in v_cis) else None,
        capsize=5,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    # Reference threshold (safety limit, e.g., 5.0%)
    target_threshold = 5.0
    ax.axhline(
        y=target_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"Target Safety Threshold ({target_threshold:.1f}%)",
    )

    # Add numeric labels on top of each bar
    max_v = max(v_means) if v_means else 100.0
    for bar, val in zip(bars, v_means):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_ylabel("Tier 1 Deadline Violation Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Tier 1 Deadline Violation Rate Across All Benchmark Models{title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, fontweight="semibold")
    ax.set_ylim(0, max(max_v * 1.15, target_threshold + 5.0))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    violation_plot_path = os.path.join(output_dir, f"violation_rate_comparison{filename_suffix}.png")
    plt.savefig(violation_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {violation_plot_path}")

    # -------------------------------------------------------------
    # Plot 2: Average Freight Cost ($)
    # -------------------------------------------------------------
    c_means = [metrics[k]["cost_mean"] for k in labels]
    c_cis = [metrics[k]["cost_ci"] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x,
        c_means,
        yerr=c_cis if any(ci > 0 for ci in c_cis) else None,
        capsize=5,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    max_c = max(c_means) if c_means else 10000.0
    for bar, val in zip(bars, c_means):
        height = bar.get_height()
        ax.annotate(
            f"${val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_ylabel("Average Freight Cost ($)", fontsize=11, fontweight="bold")
    ax.set_title(f"Average Freight Cost Across All Benchmark Models{title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, fontweight="semibold")
    ax.set_ylim(0, max_c * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    cost_plot_path = os.path.join(output_dir, f"freight_cost_comparison{filename_suffix}.png")
    plt.savefig(cost_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cost_plot_path}")


def _run_single_regime_benchmark(
    regime="nominal",
    episodes=500,
    seeds=None,
    dqn_ckpt="results/checkpoints/dqn_model.pth",
    ddqn_ckpt="results/checkpoints/ddqn_lagrangian_model.pth",
    qr_mean_ckpt="results/checkpoints/qr_dqn_mean_model.pth",
    qr_cvar_ckpt="results/checkpoints/qr_dqn_model.pth",
):
    if seeds is None:
        seeds = [0]

    # Prefer best_feasible_qr_dqn_model.pth if available
    best_feasible_path = "results/checkpoints/best_feasible_qr_dqn_model.pth"
    if (qr_cvar_ckpt == "results/checkpoints/qr_dqn_model.pth" or not os.path.exists(qr_cvar_ckpt)) and os.path.exists(best_feasible_path):
        qr_cvar_ckpt = best_feasible_path

    # Define all 7 models according to benchmark specifications
    policies = {
        "Random Masked": (lambda env: RandomMaskedAgent(env), "random"),
        "Greedy Cheapest": (lambda env: GreedyCheapestAgent(env), "heuristic"),
        "Static Threshold": (lambda env: StaticThresholdAgent(env, urgency_threshold=14.0), "heuristic"),
        "DQN": (lambda env: load_dqn_agent(dqn_ckpt), "dqn"),
        "DDQN-Lagrangian": (lambda env: load_ddqn_lagrangian_agent(ddqn_ckpt), "ddqn_lagrangian"),
        "QR-DQN Mean": (lambda env: load_qr_dqn_mean_agent(qr_mean_ckpt), "qr_dqn_mean"),
        "QR-DQN (Ours)": (lambda env: load_qr_dqn_agent(qr_cvar_ckpt), "qr_dqn"),
    }

    aggregate = {}
    metrics_summary = {}

    regime_title = (
        "Stress Disruption Regime (Adverse Shock >= 35%, Critical Deadlines 6-14d)"
        if regime == "stress"
        else "Nominal / Standard Uniform Regime"
    )

    print("\n" + "=" * 95)
    print(f"Running Monte Carlo Benchmark: {len(policies)} Models x {len(seeds)} Seed(s) x {episodes} Episodes")
    print(f"Evaluation Mode: {regime_title}")
    print("=" * 95 + "\n")

    for name, (factory, agent_type) in policies.items():
        runs = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = DubaiHumanitarianEnv()
            agent = factory(env)
            eval_res = evaluate_agent(
                agent, env, agent_type=agent_type, episodes=episodes, seed=seed, regime=regime
            )
            runs.append(eval_res)

        aggregate[name] = runs
        violation_mean, violation_ci = confidence_interval([r["violation_rate"] for r in runs])
        cost_mean, cost_ci = confidence_interval([r["average_cost"] for r in runs])
        metrics_summary[name] = {
            "violation_mean": violation_mean,
            "violation_ci": violation_ci,
            "cost_mean": cost_mean,
            "cost_ci": cost_ci,
        }

    # Rank feasible models (< 5.0% violation) by average freight cost
    feasible_models = [k for k, v in metrics_summary.items() if v["violation_mean"] <= 5.0]
    feasible_models.sort(key=lambda k: metrics_summary[k]["cost_mean"])
    rank_map = {name: f"#{i+1}" for i, name in enumerate(feasible_models)}

    # Print comprehensive CMDP comparison table
    print("\n" + "=" * 105)
    print(f"{'Model Name':<19} | {'Tier 1 Viol (%)':<20} | {'Freight Cost ($)':<22} | {'CMDP Constraint':<18} | {'Cost Rank':<10}")
    print("-" * 105)
    for name, m in metrics_summary.items():
        if len(seeds) > 1:
            v_str = f"{m['violation_mean']:5.2f}% +/- {m['violation_ci']:.2f}"
            c_str = f"${m['cost_mean']:,.2f} +/- ${m['cost_ci']:,.2f}"
        else:
            v_str = f"{m['violation_mean']:5.2f}%"
            c_str = f"${m['cost_mean']:,.2f}"
        
        is_feasible = (m["violation_mean"] <= 5.0)
        status_str = "FEASIBLE (<=5%)" if is_feasible else "VIOLATED (>5%)"
        rank_str = rank_map.get(name, "-")
        if rank_str == "#1":
            rank_str = "#1 (BEST)"
        print(f"{name:<19} | {v_str:<20} | {c_str:<22} | {status_str:<18} | {rank_str:<10}")
    print("=" * 105 + "\n")

    # Generate separate plots for this regime
    plot_suffix = " (Stress Regime)" if regime == "stress" else ""
    file_suffix = "_stress" if regime == "stress" else ""
    plot_metrics(
        list(policies.keys()),
        metrics_summary,
        output_dir="results/plots",
        title_suffix=plot_suffix,
        filename_suffix=file_suffix,
    )

    return aggregate


def run_benchmark(
    episodes=500,
    seeds=None,
    regime="all",
    dqn_ckpt="results/checkpoints/dqn_model.pth",
    ddqn_ckpt="results/checkpoints/ddqn_lagrangian_model.pth",
    qr_mean_ckpt="results/checkpoints/qr_dqn_mean_model.pth",
    qr_cvar_ckpt="results/checkpoints/qr_dqn_model.pth",
):
    if regime == "all":
        print("\n" + "#" * 95)
        print("  EXECUTING DUAL-REGIME EVALUATION: 1. STANDARD BENCHMARK & 2. STRESS TEST BENCHMARK")
        print("#" * 95)
        agg_nom = _run_single_regime_benchmark(
            regime="nominal",
            episodes=episodes,
            seeds=seeds,
            dqn_ckpt=dqn_ckpt,
            ddqn_ckpt=ddqn_ckpt,
            qr_mean_ckpt=qr_mean_ckpt,
            qr_cvar_ckpt=qr_cvar_ckpt,
        )
        agg_stress = _run_single_regime_benchmark(
            regime="stress",
            episodes=episodes,
            seeds=seeds,
            dqn_ckpt=dqn_ckpt,
            ddqn_ckpt=ddqn_ckpt,
            qr_mean_ckpt=qr_mean_ckpt,
            qr_cvar_ckpt=qr_cvar_ckpt,
        )
        return {"nominal": agg_nom, "stress": agg_stress}
    else:
        return _run_single_regime_benchmark(
            regime=regime,
            episodes=episodes,
            seeds=seeds,
            dqn_ckpt=dqn_ckpt,
            ddqn_ckpt=ddqn_ckpt,
            qr_mean_ckpt=qr_mean_ckpt,
            qr_cvar_ckpt=qr_cvar_ckpt,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all 7 benchmark agents on DubaiHumanitarianEnv.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes per seed (default: 500)")
    parser.add_argument("--seeds", type=int, default=1, help="Number of evaluation seeds (default: 1)")
    parser.add_argument("--dqn-ckpt", default="results/checkpoints/dqn_model.pth")
    parser.add_argument("--ddqn-ckpt", default="results/checkpoints/ddqn_lagrangian_model.pth")
    parser.add_argument(
        "--regime",
        choices=["nominal", "stress", "all"],
        default="all",
        help="Evaluation regime: 'nominal' (standard uniform), 'stress' (high shock >= 35%% & tight deadlines), or 'all' (default: all)",
    )
    parser.add_argument("--qr-mean-ckpt", default="results/checkpoints/qr_dqn_mean_model.pth")
    parser.add_argument("--qr-cvar-ckpt", default="results/checkpoints/qr_dqn_model.pth")

    args = parser.parse_args()
    seed_values = list(range(args.seeds))
    run_benchmark(
        episodes=args.episodes,
        seeds=seed_values,
        regime=args.regime,
        dqn_ckpt=args.dqn_ckpt,
        ddqn_ckpt=args.ddqn_ckpt,
        qr_mean_ckpt=args.qr_mean_ckpt,
        qr_cvar_ckpt=args.qr_cvar_ckpt,
    )
