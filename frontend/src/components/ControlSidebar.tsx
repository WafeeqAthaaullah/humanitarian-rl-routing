import React, { useState } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Dice5,
  Sliders,
  AlertTriangle,
  Zap,
  Clock,
  Weight,
  Layers,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type { SimulationRequest, TrajectoryStep } from "../types";

interface ControlSidebarProps {
  request: SimulationRequest;
  setRequest: React.Dispatch<React.SetStateAction<SimulationRequest>>;
  onSimulate: () => void;
  loading: boolean;
  trajectory: TrajectoryStep[];
  currentStepIndex: number;
  setCurrentStepIndex: (idx: number) => void;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;
}

export const AGENT_OPTIONS = [
  {
    id: "qr_dqn_cvar",
    name: "QR-DQN + CVaR",
    badge: "Champion",
    badgeColor: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
    desc: "Distributional RL with 10% worst-case CVaR tail risk penalty",
  },
  {
    id: "qr_dqn_mean",
    name: "QR-DQN-Mean",
    badge: "Risk-Neutral RL",
    badgeColor: "bg-cyan-500/20 text-cyan-400 border-cyan-500/40",
    desc: "Trained with 50 quantiles, selects action via mean expectation",
  },
  {
    id: "ddqn",
    name: "DDQN (Lagrangian)",
    badge: "Constrained RL",
    badgeColor: "bg-blue-500/20 text-blue-400 border-blue-500/40",
    desc: "Double DQN with static safety violation penalty (-50.0)",
  },
  {
    id: "dqn",
    name: "DQN (Standard)",
    badge: "Scalar Q-Learning",
    badgeColor: "bg-purple-500/20 text-purple-400 border-purple-500/40",
    desc: "Classic DQN maximizing expected return without risk awareness",
  },
  {
    id: "static_threshold",
    name: "Static Threshold",
    badge: "Heuristic",
    badgeColor: "bg-amber-500/20 text-amber-400 border-amber-500/40",
    desc: "Rule-based: Air if Tier 1 & deadline < 14d; else cheapest",
  },
  {
    id: "greedy",
    name: "Greedy Cheapest",
    badge: "Cost-First",
    badgeColor: "bg-red-500/20 text-red-400 border-red-500/40",
    desc: "Always dispatches ocean freight; vulnerable to bottleneck delays",
  },
  {
    id: "random",
    name: "Random Masked",
    badge: "Baseline",
    badgeColor: "bg-slate-500/20 text-slate-400 border-slate-500/40",
    desc: "Selects uniformly from legal actions defined by mask",
  },
];

export const ControlSidebar: React.FC<ControlSidebarProps> = ({
  request,
  setRequest,
  onSimulate,
  loading,
  trajectory,
  currentStepIndex,
  setCurrentStepIndex,
  isPlaying,
  setIsPlaying,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const randomizeSeed = () => {
    setRequest((prev) => ({
      ...prev,
      seed: Math.floor(Math.random() * 99999) + 1,
    }));
  };

  const handleStepBack = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(currentStepIndex - 1);
    }
  };

  const handleStepForward = () => {
    if (currentStepIndex < trajectory.length - 1) {
      setCurrentStepIndex(currentStepIndex + 1);
    }
  };

  const currentStep = trajectory[currentStepIndex];

  return (
    <div className="flex flex-col h-full bg-slate-900/90 border-r border-slate-800 p-4 space-y-5 overflow-y-auto w-full lg:w-96">
      {/* Title */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2 text-slate-200">
          <Sliders className="h-5 w-5 text-emerald-400" />
          <h2 className="font-semibold text-base">What-If Dispatch Controls</h2>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
          11-Dim CMDP
        </span>
      </div>

      {/* 1. Urgency Tier Toggle */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
          <span>Urgency Tier (u)</span>
          <span
            className={`text-xs font-semibold px-2 py-0.5 rounded ${
              request.urgency_tier === 1.0
                ? "bg-red-500/20 text-red-400 border border-red-500/30"
                : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
            }`}
          >
            {request.urgency_tier === 1.0 ? "Tier 1: Critical Emergency" : "Tier 2: Standard Relief"}
          </span>
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => setRequest((p) => ({ ...p, urgency_tier: 1.0 }))}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all border ${
              request.urgency_tier === 1.0
                ? "bg-red-500/20 border-red-500/60 text-red-300 shadow-sm shadow-red-950"
                : "bg-slate-800/60 border-slate-700/60 text-slate-400 hover:text-slate-200"
            }`}
          >
            🚨 Tier 1 (Critical)
          </button>
          <button
            type="button"
            onClick={() => setRequest((p) => ({ ...p, urgency_tier: 2.0 }))}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all border ${
              request.urgency_tier === 2.0
                ? "bg-blue-500/20 border-blue-500/60 text-blue-300 shadow-sm shadow-blue-950"
                : "bg-slate-800/60 border-slate-700/60 text-slate-400 hover:text-slate-200"
            }`}
          >
            📦 Tier 2 (Standard)
          </button>
        </div>
      </div>

      {/* 2. Shipment Weight Slider */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="text-slate-300 flex items-center space-x-1.5">
            <Weight className="h-3.5 w-3.5 text-cyan-400" />
            <span>Shipment Weight (w)</span>
          </span>
          <span className="font-mono text-cyan-400 font-semibold">
            {request.shipment_weight.toFixed(1)} tons
          </span>
        </div>
        <input
          type="range"
          min="0.5"
          max="50.0"
          step="0.5"
          value={request.shipment_weight}
          onChange={(e) =>
            setRequest((p) => ({
              ...p,
              shipment_weight: parseFloat(e.target.value),
            }))
          }
          className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>0.5t (Light / Meds)</span>
          <span>Max Air Cap: 50.0t</span>
        </div>
      </div>

      {/* 3. Deadline Remaining Slider */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="text-slate-300 flex items-center space-x-1.5">
            <Clock className="h-3.5 w-3.5 text-amber-400" />
            <span>Deadline Remaining (d_rem)</span>
          </span>
          <span className="font-mono text-amber-400 font-semibold">
            {request.deadline_remaining.toFixed(1)} days
          </span>
        </div>
        <input
          type="range"
          min="5.0"
          max="30.0"
          step="0.5"
          value={request.deadline_remaining}
          onChange={(e) =>
            setRequest((p) => ({
              ...p,
              deadline_remaining: parseFloat(e.target.value),
            }))
          }
          className="w-full accent-amber-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>5.0d (Urgent)</span>
          <span>30.0d (Generous)</span>
        </div>
      </div>

      {/* 4. Disruption Shock Probability Slider */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="text-slate-300 flex items-center space-x-1.5">
            <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
            <span>Disruption Shock Prob (S_disrupt)</span>
          </span>
          <span className="font-mono text-red-400 font-semibold">
            {(request.disruption_shock_prob * 100).toFixed(0)}%
          </span>
        </div>
        <input
          type="range"
          min="0.0"
          max="0.5"
          step="0.05"
          value={request.disruption_shock_prob}
          onChange={(e) =>
            setRequest((p) => ({
              ...p,
              disruption_shock_prob: parseFloat(e.target.value),
            }))
          }
          className="w-full accent-red-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>0% (Calm Sea Lanes)</span>
          <span>50% (High Bottleneck Shock)</span>
        </div>
      </div>

      {/* 5. Random Seed */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-300 flex items-center space-x-1.5">
            <Dice5 className="h-3.5 w-3.5 text-purple-400" />
            <span>Simulation Random Seed</span>
          </span>
          <button
            type="button"
            onClick={randomizeSeed}
            className="text-[11px] text-purple-400 hover:text-purple-300 flex items-center space-x-1 font-mono"
          >
            <span>Roll Seed</span>
          </button>
        </div>
        <div className="flex space-x-2">
          <input
            type="number"
            value={request.seed ?? 42}
            onChange={(e) =>
              setRequest((p) => ({
                ...p,
                seed: parseInt(e.target.value) || 0,
              }))
            }
            className="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-purple-500"
          />
          <button
            type="button"
            onClick={randomizeSeed}
            className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300"
            title="Randomize Seed"
          >
            <Dice5 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* 6. Active Agent Selector */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-slate-300 flex items-center space-x-1.5">
          <Zap className="h-3.5 w-3.5 text-emerald-400" />
          <span>Active Agent Architecture</span>
        </label>
        <div className="space-y-1.5">
          {AGENT_OPTIONS.map((agent) => (
            <button
              key={agent.id}
              type="button"
              onClick={() => setRequest((p) => ({ ...p, agent_id: agent.id }))}
              className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all flex flex-col justify-between ${
                request.agent_id === agent.id
                  ? "bg-emerald-500/10 border-emerald-500/50 text-white shadow-sm"
                  : "bg-slate-800/40 border-slate-800/80 text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
              }`}
            >
              <div className="flex items-center justify-between w-full">
                <span className="font-semibold text-slate-100">{agent.name}</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${agent.badgeColor}`}
                >
                  {agent.badge}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                {agent.desc}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Parameters Accordion */}
      <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-950/40">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-3 py-2 text-xs font-medium text-slate-400 hover:text-slate-200 flex items-center justify-between"
        >
          <span>Extended State Variables (Volume, Capacities)</span>
          {showAdvanced ? (
            <ChevronUp className="h-3.5 w-3.5" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5" />
          )}
        </button>
        {showAdvanced && (
          <div className="p-3 border-t border-slate-800 space-y-3 text-xs">
            <div>
              <div className="flex justify-between text-slate-400 mb-1">
                <span>Cargo Volume (v):</span>
                <span className="font-mono text-white">{request.volume} m³</span>
              </div>
              <input
                type="range"
                min="5"
                max="100"
                value={request.volume}
                onChange={(e) =>
                  setRequest((p) => ({
                    ...p,
                    volume: parseFloat(e.target.value),
                  }))
                }
                className="w-full accent-emerald-500 h-1 bg-slate-800 rounded"
              />
            </div>
            <div>
              <div className="flex justify-between text-slate-400 mb-1">
                <span>Dest. Buffer Inventory (I_dest):</span>
                <span className="font-mono text-white">
                  {request.dest_inventory} days
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="50"
                value={request.dest_inventory}
                onChange={(e) =>
                  setRequest((p) => ({
                    ...p,
                    dest_inventory: parseFloat(e.target.value),
                  }))
                }
                className="w-full accent-cyan-500 h-1 bg-slate-800 rounded"
              />
            </div>
          </div>
        )}
      </div>

      {/* Primary Simulate Dispatch Button */}
      <button
        type="button"
        onClick={onSimulate}
        disabled={loading}
        className={`w-full py-3 px-4 rounded-xl font-semibold text-sm tracking-wide text-white transition-all shadow-lg flex items-center justify-center space-x-2 ${
          loading
            ? "bg-slate-700 cursor-not-allowed"
            : "bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 shadow-emerald-950/60 active:scale-[0.98]"
        }`}
      >
        {loading ? (
          <>
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            <span>Simulating Echelons...</span>
          </>
        ) : (
          <>
            <Zap className="h-4 w-4" />
            <span>Simulate Dispatch</span>
          </>
        )}
      </button>

      {/* Step Playback Controls */}
      {trajectory.length > 0 && (
        <div className="pt-2 border-t border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-300 font-medium flex items-center space-x-1.5">
              <Layers className="h-3.5 w-3.5 text-emerald-400" />
              <span>Playback Timeline</span>
            </span>
            <span className="font-mono text-emerald-400 font-semibold">
              Step {currentStepIndex + 1} of {trajectory.length}
            </span>
          </div>

          <div className="flex items-center justify-center space-x-2 bg-slate-950/60 p-2 rounded-xl border border-slate-800">
            <button
              onClick={() => setCurrentStepIndex(0)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              title="Reset to Step 1"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            <button
              onClick={handleStepBack}
              disabled={currentStepIndex === 0}
              className={`p-1.5 rounded-lg ${
                currentStepIndex === 0
                  ? "text-slate-600 cursor-not-allowed"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              }`}
              title="Step Back"
            >
              <SkipBack className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="px-3 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 flex items-center space-x-1 text-xs font-semibold"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? (
                <>
                  <Pause className="h-3.5 w-3.5 fill-current" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-current" />
                  <span>Play</span>
                </>
              )}
            </button>
            <button
              onClick={handleStepForward}
              disabled={currentStepIndex === trajectory.length - 1}
              className={`p-1.5 rounded-lg ${
                currentStepIndex === trajectory.length - 1
                  ? "text-slate-600 cursor-not-allowed"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              }`}
              title="Step Forward"
            >
              <SkipForward className="h-4 w-4" />
            </button>
          </div>

          {/* Current Step Inspector Card */}
          {currentStep && (
            <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between text-slate-300 font-sans">
                <span className="font-semibold text-slate-100">
                  {currentStep.action_name}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold ${
                    currentStep.is_pivot
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                      : "bg-slate-800 text-slate-300"
                  }`}
                >
                  {currentStep.transport_mode}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px] pt-1 border-t border-slate-800/60">
                <div>
                  <span className="text-slate-500 block">Step Freight:</span>
                  <span className="text-emerald-400 font-semibold">
                    ${currentStep.freight_cost.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block">Leg Transit Time:</span>
                  <span className="text-amber-400 font-semibold">
                    +{currentStep.days_passed.toFixed(1)} days
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block">Cumulative Days:</span>
                  <span className="text-slate-200">
                    {currentStep.days_elapsed.toFixed(1)} days
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block">Deadline Slack:</span>
                  <span
                    className={
                      currentStep.deadline_remaining >= 0
                        ? "text-emerald-400"
                        : "text-red-400 font-bold"
                    }
                  >
                    {currentStep.deadline_remaining.toFixed(1)} days
                  </span>
                </div>
              </div>
              {currentStep.disruption_triggered && (
                <div className="p-2 rounded bg-red-950/40 border border-red-500/40 text-red-300 text-[11px] font-sans flex items-center space-x-1.5">
                  <AlertTriangle className="h-4 w-4 shrink-0 text-red-400" />
                  <span>Maritime bottleneck disruption triggered on this leg!</span>
                </div>
              )}
              {currentStep.is_pivot && (
                <div className="p-2 rounded bg-amber-950/40 border border-amber-500/40 text-amber-300 text-[11px] font-sans flex items-center space-x-1.5">
                  <Zap className="h-4 w-4 shrink-0 text-amber-400" />
                  <span>Emergency Intermodal Air Pivot executed at Checkpoint!</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
