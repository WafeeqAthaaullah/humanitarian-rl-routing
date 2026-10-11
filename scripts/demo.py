import sys
import os
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.dubai_humanitarian_env import DubaiHumanitarianEnv
from agents.qr_dqn_agent import QRDQNAgent

action_map = {0: "AIR_DIRECT", 1: "OCEAN_TO_HUB", 2: "LAND_TO_HUB", 3: "SWITCH_TO_AIR", 4: "CONTINUE_NOMINAL"}

def run_live_demo():
    env = DubaiHumanitarianEnv()
    agent = QRDQNAgent()
    agent.network.load_state_dict(torch.load("results/checkpoints/qr_dqn_model.pth", weights_only=True))
    agent.network.eval()
    
    # Force a challenging scenario for the demo
    state, _ = env.reset()
    state[0] = 1.0  # Tier 1 Critical
    state[3] = 10.0 # Very tight deadline
    state[10] = 0.3 # High disruption probability
    env.state = state
    
    print("\n" + "="*50)
    print("DUBAI HUMANITARIAN LIVE DISPATCH DEMO")
    print("="*50)
    print(f"Cargo: Tier {int(state[0])} | Deadline: {state[3]} days | Disruption Risk: {state[10]*100}%")
    print("-" * 50)
    
    while True:
        time.sleep(1.5) # Pause for dramatic effect during presentation
        loc = "Dubai Hub" if state[4] == 0.0 else "Regional Checkpoint"
        print(f"Current Location: {loc}")
        
        action = agent.act_cvar(state, epsilon=0.0)
        print(f"Agent selects: {action_map[action]}")
        
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print("-" * 50)
            status = "SUCCESS (On Time)" if info.get("deadline_hit") else "CATASTROPHIC FAILURE (Missed Deadline)"
            print(f"Episode Terminated: {status}")
            print(f"Total Days Taken: {10.0 - state[3]:.1f}")
            break

if __name__ == "__main__":
    run_live_demo()