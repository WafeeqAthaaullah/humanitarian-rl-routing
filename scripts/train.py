import sys
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dubai_humanitarian_env import DubaiHumanitarianEnv
from agents.qr_dqn_agent import QRDQNAgent
from config import (
    TRAIN_EPISODES, LEARNING_RATE,
    EPSILON_START, EPSILON_DECAY, EPSILON_MIN,
    BATCH_SIZE, TARGET_UPDATE_FREQ, LAMBDA_LR,
)

def train_agent():
    env = DubaiHumanitarianEnv()
    agent = QRDQNAgent()
    agent.setup_training(lr=LEARNING_RATE)
    
    epsilon = EPSILON_START
    
    # Lagrangian Multiplier for safety constraint
    lambda_penalty = 0.0 
    
    # Metrics tracking for curves and analysis
    history = {
        "episode": [],
        "reward": [],
        "loss": [],
        "epsilon": [],
        "lambda_penalty": [],
        "violation": [],
    }

    print(f"Starting training for {TRAIN_EPISODES} episodes...")
    
    for ep in range(TRAIN_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        loss_sum = 0
        steps = 0
        
        # Capture initial urgency at episode start (avoids using stale state later)
        is_tier1 = (state[0] == 1.0)
        violated = False
        
        while True:
            # 1. Select action
            action = agent.act_cvar(state, epsilon)
            
            # 2. Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Apply current Lagrangian penalty if critical deadline violated
            if terminated and info.get("deadline_hit") is False and is_tier1:
                reward -= lambda_penalty
                violated = True
                
            # 3. Store in memory
            agent.memory.push(state, action, reward, next_state, terminated)
            
            # 4. Train
            loss = agent.train_step(batch_size=BATCH_SIZE)
            loss_sum += loss
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
                
        # Update exploration rate
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Update target network periodically
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            
        # Update Lagrangian multiplier based on safety violations
        if violated:
            lambda_penalty += LAMBDA_LR * 1.0  # Increase penalty for violation
        else:
            lambda_penalty = max(0.0, lambda_penalty - LAMBDA_LR * 1.0)  # Decay penalty if compliant
        
        # Record metrics
        avg_loss = loss_sum / max(1, steps)
        history["episode"].append(ep)
        history["reward"].append(episode_reward)
        history["loss"].append(avg_loss)
        history["epsilon"].append(epsilon)
        history["lambda_penalty"].append(lambda_penalty)
        history["violation"].append(1 if violated else 0)
        
        if ep % 100 == 0:
            recent_violations = np.mean(history["violation"][-100:]) * 100
            print(f"Episode {ep:4d} | Reward: {episode_reward:6.2f} | Loss: {avg_loss:.4f} | "
                  f"Epsilon: {epsilon:.2f} | Lambda: {lambda_penalty:.2f} | Violation Rate (100-ep): {recent_violations:.1f}%")
    
    # Save model weights
    os.makedirs("results/checkpoints", exist_ok=True)
    model_path = "results/checkpoints/qr_dqn_model.pth"
    torch.save(agent.network.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save training logs to CSV
    os.makedirs("results/logs", exist_ok=True)
    csv_path = "results/logs/training_metrics.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "loss", "epsilon", "lambda_penalty", "violation"])
        for i in range(len(history["episode"])):
            writer.writerow([
                history["episode"][i],
                history["reward"][i],
                history["loss"][i],
                history["epsilon"][i],
                history["lambda_penalty"][i],
                history["violation"][i],
            ])
    print(f"Training metrics saved to {csv_path}")

    # Plot & save training curves
    plot_training_curves(history)

def plot_training_curves(history, window=50):
    os.makedirs("results/plots", exist_ok=True)
    
    episodes = history["episode"]
    rewards = np.array(history["reward"])
    losses = np.array(history["loss"])
    lambdas = np.array(history["lambda_penalty"])
    violations = np.array(history["violation"])
    
    def moving_avg(arr, w):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("QR-DQN Training Dynamics (Dubai Humanitarian Routing)", fontsize=16, fontweight='bold')

    # Subplot 1: Episode Reward & Moving Average
    axs[0, 0].plot(episodes, rewards, alpha=0.2, color="gray", label="Episode Reward")
    if len(rewards) >= window:
        axs[0, 0].plot(episodes[window-1:], moving_avg(rewards, window), color="tab:blue", label=f"{window}-Ep Moving Avg")
    axs[0, 0].set_title("Reward Progression")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Quantile Huber Loss
    axs[0, 1].plot(episodes, losses, alpha=0.3, color="gray", label="Loss")
    if len(losses) >= window:
        axs[0, 1].plot(episodes[window-1:], moving_avg(losses, window), color="tab:orange", label=f"{window}-Ep Moving Avg")
    axs[0, 1].set_title("Quantile Huber Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Deadline Violation Rate
    if len(violations) >= window:
        viol_rate = moving_avg(violations, window) * 100
        axs[1, 0].plot(episodes[window-1:], viol_rate, color="tab:red", label=f"{window}-Ep Violation Rate (%)")
    else:
        axs[1, 0].plot(episodes, violations * 100, color="tab:red", label="Violation (%)")
    axs[1, 0].set_title("Deadline Violation Rate")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Violation Rate (%)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Dual Axis - Lambda Multiplier and Epsilon
    ax_lam = axs[1, 1]
    ax_eps = ax_lam.twinx()
    
    line1 = ax_lam.plot(episodes, lambdas, color="tab:purple", label="Lambda (Penalty Multiplier)")
    line2 = ax_eps.plot(episodes, history["epsilon"], color="tab:green", linestyle="--", label="Epsilon (Exploration)")
    
    ax_lam.set_title("Lagrangian Multiplier & Exploration")
    ax_lam.set_xlabel("Episode")
    ax_lam.set_ylabel("Lambda", color="tab:purple")
    ax_eps.set_ylabel("Epsilon", color="tab:green")
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_lam.legend(lines, labels, loc="upper right")
    ax_lam.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "results/plots/training_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    train_agent()