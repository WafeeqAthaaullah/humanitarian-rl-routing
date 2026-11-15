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
├── backend/                        # Modular FastAPI backend
│   ├── main.py                    # REST API server & static asset mount
│   ├── schemas.py                 # Pydantic models for simulation & showdown
│   └── inference.py               # Checkpoint caching, CVaR inference & simulation runner
├── frontend/                       # Interactive React + TypeScript + Tailwind CSS web app
│   ├── src/
│   │   ├── components/
│   │   │   ├── ControlSidebar.tsx         # What-If scenario controls & timeline playback
│   │   │   ├── RouteVisualizer.tsx        # Multi-echelon flow diagram & intermodal pivot animation
│   │   │   ├── QuantileRiskInspector.tsx  # Plotly 50-quantile distribution & 5% CVaR tail inspector
│   │   │   ├── AgentShowdown.tsx          # 7-model synchronous benchmark comparison
│   │   │   ├── TrainingDiagnostics.tsx    # Training telemetry curves
│   │   │   ├── Navbar.tsx                 # Operational command center header
│   │   │   └── PlotlyChart.tsx            # Responsive Plotly container
│   │   ├── api.ts                         # Typed API client
│   │   ├── types.ts                       # TypeScript interfaces
│   │   └── App.tsx                        # Main command center layout
│   └── package.json
├── run_app.sh                      # Unified Unix/Mac bash launcher script
├── run_app.bat                     # Windows one-click command center launcher
├── config.py                       # Centralized environment and agent hyperparameters
├── requirements.txt                # Python package dependencies
├── envs/
│   ├── __init__.py
│   └── dubai_humanitarian_env.py   # Custom Gymnasium simulation environment
├── agents/
│   ├── heuristics.py              # Greedy and Threshold baseline agents
│   ├── random_agent.py            # Random masked action baseline
│   ├── dqn_agent.py               # Standard scalar Deep Q-Network baseline
│   ├── ddqn_lagrangian_agent.py   # Double DQN with Lagrangian safety penalty
│   ├── qr_dqn_mean_agent.py       # QR-DQN with risk-neutral mean action selection
│   └── qr_dqn_agent.py            # QR-DQN network with risk-averse CVaR policy
├── scripts/
│   ├── train.py                    # Training loop with Lagrangian safety updates
│   ├── evaluate_model.py           # Multi-agent benchmark evaluation
│   └── demo.py                     # CLI trace of high-risk cargo dispatch
├── results/
│   ├── checkpoints/               # Trained neural network weights (.pth)
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

## 🌐 Web Application (Command Center)

An interactive, dark-mode operational command center built with **FastAPI** (backend) and **React + TypeScript + Tailwind CSS + Plotly.js** (frontend) to explore the 3-echelon environment, inspect distributional tail risk, and run synchronous 7-model showdowns.

### Quick Start (One Command)
- **Linux / macOS**:
  ```bash
  chmod +x run_app.sh
  ./run_app.sh
  ```
- **Windows**:
  Double-click `run_app.bat` or run in terminal:
  ```powershell
  .\run_app.bat
  ```

### Manual Execution

#### 1. Start FastAPI Backend (Port 8000)
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Unified Web Application: `http://127.0.0.1:8000/`

#### 2. Start React Dev Server (Port 5173, optional for development)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

### Core Interactive Features

1. **Interactive "What-If" Dispatch Simulator & Playback Timeline**:
   - Sidebar matching the 11-dimensional state space: Urgency Tier (Tier 1 vs. Tier 2), Shipment Weight (0.5–50 t), Deadline Remaining (5–30 days), and Disruption Shock Probability (0%–50%).
   - Select any of the 7 models and click **"Simulate Dispatch"** to execute a seeded episode live.
   - Step-by-step playback controls: Play/Pause, Step Back/Forward, and step inspector card.

2. **Multi-Echelon Route & Intermodal Pivot Visualizer**:
   - Dynamic SVG diagram illustrating the 3 echelons: **Dubai Origin $\rightarrow$ Regional Checkpoint $\rightarrow$ Disaster Site**.
   - Distinct paths for Direct Air, Ocean Leg 1, Overland Leg 1, Nominal Leg 2, and the **Emergency Intermodal Air Pivot (`SWITCH_TO_AIR`)**.
   - Live visual disruption alerts when maritime delays trigger and animated highlight of adaptive intermodal recovery.

3. **Real-Time 50-Quantile & 5% CVaR Risk Inspector (`Plotly.js`)**:
   - Visualizes all 50 predicted return quantiles ($Z_1 \dots Z_{50}$) for each candidate action.
   - Red shaded risk zone highlights the worst $\alpha = 5\%$ tail ($Z_1 \dots Z_3$) with a vertical cutoff.
   - Side-by-side callout reveals why `QR-DQN + CVaR` rejects ocean freight when disruption risk spikes while standard `DQN` and `QR-DQN-Mean` get trapped by the average.

4. **Live 7-Model Synchronous Showdown**:
   - Simultaneously benchmarks **Greedy, Static Threshold, Random, DQN, DDQN, QR-DQN-Mean, and QR-DQN + CVaR** on the exact same seeded disruption scenario.
   - Side-by-side comparison table featuring delivery badges (**`DELIVERED`** vs. **`DEADLINE VIOLATED`**), total freight cost, elapsed days, intermodal pivot status, and route sequence.

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
