import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add repo root to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.schemas import (
    SimulationRequest,
    SimulationResponse,
    ShowdownRequest,
    ShowdownResponse,
    TrainingMetricsResponse,
    QuantileAnalysis,
)
from backend.inference import (
    ModelRegistry,
    AGENT_METADATA,
    ACTION_NAMES,
    ACTION_DISPLAY_NAMES,
    run_single_simulation,
    run_showdown,
    load_training_metrics,
)

app = FastAPI(
    title="Dubai Humanitarian RL Routing Operational Command Center",
    description="Interactive backend for multi-echelon humanitarian logistics, risk-averse QR-DQN with CVaR, and 7-model showdown.",
    version="1.0.0",
)

# Enable CORS for local React/Vite development or any host
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_models():
    """Pre-warm checkpoints in memory on startup."""
    ModelRegistry.get_instance()


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Dubai Humanitarian RL Routing Engine",
        "models_available": list(AGENT_METADATA.keys()),
    }


@app.get("/api/models")
def get_models():
    """Return all 7 evaluated model configurations and descriptions."""
    return {
        "models": AGENT_METADATA,
        "action_names": ACTION_NAMES,
        "action_display_names": ACTION_DISPLAY_NAMES,
    }


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate_dispatch(request: SimulationRequest):
    """
    Run an interactive multi-echelon dispatch simulation for the specified agent.
    Returns step-by-step trajectory, state vectors, action masks, freight cost,
    delay dynamics, maritime disruption detection, and quantile risk analysis.
    """
    try:
        response = run_single_simulation(
            agent_id=request.agent_id,
            urgency_tier=request.urgency_tier,
            shipment_weight=request.shipment_weight,
            deadline_remaining=request.deadline_remaining,
            disruption_shock_prob=request.disruption_shock_prob,
            seed=request.seed if request.seed is not None else 42,
            volume=request.volume if request.volume is not None else 30.0,
            dest_inventory=request.dest_inventory if request.dest_inventory is not None else 20.0,
            c_air=request.c_air if request.c_air is not None else 1.0,
            c_ocean=request.c_ocean if request.c_ocean is not None else 1.0,
            c_land=request.c_land if request.c_land is not None else 1.0,
        )
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/showdown", response_model=ShowdownResponse)
def run_model_showdown(request: ShowdownRequest):
    """
    Execute the exact same seeded disruption scenario simultaneously across all 7 models:
    Greedy, Static Threshold, Random, DQN, DDQN, QR-DQN-Mean, and QR-DQN + CVaR.
    """
    try:
        response = run_showdown(
            urgency_tier=request.urgency_tier,
            shipment_weight=request.shipment_weight,
            deadline_remaining=request.deadline_remaining,
            disruption_shock_prob=request.disruption_shock_prob,
            seed=request.seed if request.seed is not None else 42,
            volume=request.volume if request.volume is not None else 30.0,
            dest_inventory=request.dest_inventory if request.dest_inventory is not None else 20.0,
        )
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training-metrics", response_model=TrainingMetricsResponse)
def get_training_metrics(downsample: int = 25):
    """Serve parsed training telemetry from results/logs/training_metrics.csv."""
    try:
        return load_training_metrics(downsample_factor=downsample)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quantile-inspection", response_model=QuantileAnalysis)
def inspect_state_quantiles(state: list[float]):
    """Compute and return 5x50 quantile matrix, CVaR at 5% and 10%, and mean Q-values for any state."""
    import numpy as np
    try:
        registry = ModelRegistry.get_instance()
        arr = np.array(state, dtype=np.float32)
        if len(arr) != 11:
            raise ValueError(f"Expected 11-dimensional state vector, received {len(arr)}")
        return registry.extract_quantiles_analysis(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Mount frontend static files if production build exists
frontend_dist = os.path.join(REPO_ROOT, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
