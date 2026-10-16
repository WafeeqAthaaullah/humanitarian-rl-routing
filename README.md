# Distributional RL for Humanitarian Transport-Mode Selection
**Partnership Context:** Dubai Humanitarian | **Focus:** Multi-Echelon Logistics under Disruption  
**Course:** CSCI218 Project

---

## 📌 Project Overview
Dispatching critical humanitarian aid requires balancing tight financial budgets with strict delivery deadlines. Surface modes (ocean and land) offer lower freight costs but suffer from fat-tailed transit delays when disruptions strike (e.g., port congestion, customs bottlenecks). Air transport guarantees rapid transit at significantly higher expense.

This project implements a **Constrained Markov Decision Process (CMDP)** and trains a risk-sensitive **Quantile Regression Deep Q-Network (QR-DQN)** using **Conditional Value at Risk (CVaR)** action selection to reliably meet emergency deadlines while minimizing freight expenditure.

---

## 🚀 Key Features
1. **Gymnasium Environment (`DubaiHumanitarianEnv`)**:
   - **State Space (11D)**: Cargo urgency, volume, weight, remaining deadline, transit location, active mode, multi-modal modal capacities, destination inventory buffer, and shock probability.
   - **Action Space (5 Discrete Modes)**:
     - `0: AIR_DIRECT`
     - `1: OCEAN_TO_HUB`
     - `2: LAND_TO_HUB`
     - `3: SWITCH_TO_AIR`
     - `4: CONTINUE_NOMINAL`
   - **Stochastic Delays**: Normal distribution for nominal routes and fat-tailed Log-Normal disruption mixture models for maritime bottlenecks.
2. **Distributional RL (QR-DQN)**:
   - Predicts 50 return quantiles per action rather than single scalar Q-values.
   - Action selection uses **CVaR ($\alpha=0.10$)** to optimize for worst-case outcomes.
   - Robust training with Quantile Huber loss, gradient clipping, observation normalization, and target network stabilization.
3. **Constrained Optimization**:
   - Adaptive **Lagrangian multiplier** dynamically penalizes deadline violations on critical Tier-1 cargo.
4. **Benchmarking & Heuristics**:
   - Evaluated against `GreedyCheapestAgent` and `StaticThresholdAgent`.

---

## 📂 Project Structure
```text
humanitarian-rl-routing/
├── config.py                       # Centralized environment and agent hyperparameters
├── requirements.txt                # Python package dependencies
├── envs/
│   ├── __init__.py
│   └── dubai_humanitarian_env.py   # Custom Gymnasium simulation environment
├── agents/
│   ├── __init__.py
│   ├── heuristics.py              # Greedy and Threshold baseline agents
│   └── qr_dqn_agent.py            # QR-DQN network, CVaR policy & ReplayBuffer
├── scripts/
│   ├── train.py                    # Training loop with Lagrangian safety updates & curve plotting
│   ├── evaluate_model.py           # Multi-agent benchmark evaluation & comparison plot
│   └── demo.py                     # Interactive CLI trace of high-risk cargo dispatch
├── results/
│   ├── checkpoints/               # Saved model weights (qr_dqn_model.pth)
│   ├── logs/                      # Training history metrics (training_metrics.csv)
│   └── plots/                     # Evaluation and training curves (PNGs)
└── project_demonstration.ipynb     # Interactive Jupyter demonstration notebook
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.9+ installed
- Virtual environment recommended

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Train the Agent
Run the training pipeline (trains the agent, saves checkpoint to `results/checkpoints/qr_dqn_model.pth`, logs metrics, and generates training plots):
```bash
python scripts/train.py
```

### Benchmark Agents
Evaluate QR-DQN against repeated-seed baselines over 500 Monte Carlo episodes:
```bash
python scripts/evaluate_model.py
```
The benchmark reports 95% confidence intervals, Tier 1 violation rates, explicit
freight costs, deadline slack, and invalid-action rates. It includes Always-Air,
Always-Ocean, Greedy-Cheapest, Static Threshold, and QR-DQN baselines. Use
`--episodes` and `--seeds` to control experiment size.

```bash
python scripts/evaluate_model.py --episodes 500 --seeds 10
```
*Outputs comparative statistics and saves a plot to `results/plots/performance_comparison.png`.*

### Run Live Interactive Demo
Simulate a single dispatch scenario under adverse shock probability:
```bash
python scripts/demo.py
```

### Interactive Notebook
Open `project_demonstration.ipynb` in Jupyter Notebook or VS Code to step through problem formulation, evaluation charts, and live execution traces.

---

## 📊 Evaluation & Metrics
- **Deadline Violation Rate (%)**: Proportion of Tier-1 critical shipments missing deadline.
- **Average Freight Cost ($)**: Realized transportation cost per shipment.
- **Safety vs. Cost Trade-off**: Compares conservative threshold heuristic, cost-blind greedy dispatch, and risk-aware QR-DQN.
- **Reproducibility**: Training accepts a seed, records configuration metadata in the checkpoint, and saves a best-feasible checkpoint when the Tier 1 constraint is satisfied.
- **Action masking**: Agents receive legal, capacity-feasible action masks; random exploration cannot deliberately select impossible transitions.
