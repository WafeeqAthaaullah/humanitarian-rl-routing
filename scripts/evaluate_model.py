import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv
from agents.heuristics import GreedyCheapestAgent, StaticThresholdAgent
from agents.qr_dqn_agent import QRDQNAgent

def evaluate_agent(agent, env, agent_type="heuristic", episodes=500):
    total_cost = 0.0
    violations = 0
    tier1_count = 0
    
    for _ in range(episodes):
        state, _ = env.reset()
        is_tier1 = (state[0] == 1.0)
        if is_tier1:
            tier1_count += 1
            
        while True:
            if agent_type == "rl":
                # RL agent acts greedily (epsilon=0) during evaluation
                action = agent.act_cvar(state, epsilon=0.0)
            else:
                action = agent.act(state)
                
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                # Approximate cost (reversing the normalization we did in env)
                if info.get("reason") == "invalid_action":
                    total_cost += 100000 # Massive penalty for failure
                    if is_tier1: violations += 1
                else:
                    # Subtract the safety penalty if it missed the deadline to isolate pure cost
                    raw_reward = abs(reward)
                    if not info.get("deadline_hit") and is_tier1:
                        raw_reward -= 10.0
                        violations += 1
                    total_cost += (raw_reward * 10000)
                break
                
    violation_rate = (violations / max(1, tier1_count)) * 100
    avg_cost = total_cost / episodes
    return violation_rate, avg_cost

if __name__ == "__main__":
    env = DubaiHumanitarianEnv()
    
    # 1. Initialize Agents
    greedy = GreedyCheapestAgent(env)
    threshold = StaticThresholdAgent(env, urgency_threshold=12.0)
    
    rl_agent = QRDQNAgent()
    # Load the trained weights
    rl_agent.network.load_state_dict(torch.load("results/checkpoints/qr_dqn_model.pth", weights_only=True))
    rl_agent.network.eval()
    
    print("Evaluating agents over 500 episodes...")
    v_greedy, c_greedy = evaluate_agent(greedy, env, "heuristic")
    v_thresh, c_thresh = evaluate_agent(threshold, env, "heuristic")
    v_rl, c_rl = evaluate_agent(rl_agent, env, "rl")
    
    print(f"Greedy    | Violation Rate: {v_greedy:.1f}% | Avg Cost: ${c_greedy:,.2f}")
    print(f"Threshold | Violation Rate: {v_thresh:.1f}% | Avg Cost: ${c_thresh:,.2f}")
    print(f"QR-DQN    | Violation Rate: {v_rl:.1f}% | Avg Cost: ${c_rl:,.2f}")
    
    # 2. Plot Results
    labels = ['Greedy', 'Threshold', 'QR-DQN (Ours)']
    violations = [v_greedy, v_thresh, v_rl]
    costs = [c_greedy, c_thresh, c_rl]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_ylabel('Deadline Violation Rate (%)', color=color, fontweight='bold')
    bars = ax1.bar([x - 0.2 for x in range(3)], violations, width=0.4, color=color, label='Violation Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average Freight Cost ($)', color=color, fontweight='bold')
    ax2.bar([x + 0.2 for x in range(3)], costs, width=0.4, color=color, alpha=0.7, label='Cost')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Agent Performance: Safety vs. Financial Cost')
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/performance_comparison.png")
    print("Plot saved to results/plots/performance_comparison.png")