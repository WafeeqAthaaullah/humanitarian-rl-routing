import React from "react";
import {
  Ship,
  Plane,
  AlertTriangle,
  Zap,
  CheckCircle2,
  XCircle,
  MapPin,
  Clock,
  DollarSign,
  Package,
} from "lucide-react";
import type { SimulationResponse, TrajectoryStep } from "../types";

interface RouteVisualizerProps {
  simulation: SimulationResponse | null;
  currentStepIndex: number;
}

export const RouteVisualizer: React.FC<RouteVisualizerProps> = ({
  simulation,
  currentStepIndex,
}) => {
  if (!simulation || simulation.trajectory.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[460px] bg-slate-950/60 rounded-2xl border border-slate-800/80 p-8 text-center">
        <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-4 animate-pulse">
          <Plane className="h-8 w-8" />
        </div>
        <h3 className="text-lg font-semibold text-slate-200">
          Ready for Dispatch Simulation
        </h3>
        <p className="text-xs text-slate-400 max-w-md mt-2">
          Adjust scenario parameters in the left sidebar (Urgency Tier, Weight,
          Deadline, Disruption Shock) and click{" "}
          <strong className="text-emerald-400">"Simulate Dispatch"</strong> to
          visualize the live multi-echelon journey and dynamic intermodal pivots.
        </p>
      </div>
    );
  }

  const currentStep: TrajectoryStep = simulation.trajectory[currentStepIndex] || simulation.trajectory[0];
  const isFinalStep = currentStepIndex === simulation.trajectory.length - 1;
  const isDelivered = simulation.status === "Delivered";

  // Check which routes have been traversed up to current step
  const executedSteps = simulation.trajectory.slice(0, currentStepIndex + 1);
  const directAirActive = executedSteps.some((s) => s.action === 0);
  const oceanLeg1Active = executedSteps.some((s) => s.action === 1);
  const landLeg1Active = executedSteps.some((s) => s.action === 2);
  const pivotAirActive = executedSteps.some((s) => s.action === 3);
  const continueNominalActive = executedSteps.some((s) => s.action === 4);

  // Disruption on current step or any step so far
  const disruptionTriggeredNow = currentStep.disruption_triggered;
  const anyDisruption = executedSteps.some((s) => s.disruption_triggered);

  // Active step transport mode indicator
  const activeAction = currentStep.action;

  return (
    <div className="flex flex-col h-full bg-slate-950/80 rounded-2xl border border-slate-800/80 overflow-hidden shadow-2xl relative">
      {/* Top Status & KPI Header */}
      <div className="p-4 bg-slate-900/80 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <div
            className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center space-x-1.5 border shadow-sm ${
              isDelivered
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                : "bg-red-500/20 text-red-300 border-red-500/40"
            }`}
          >
            {isDelivered ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                <span>DELIVERED • IN SAFETY BUDGET</span>
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4 text-red-400" />
                <span>DEADLINE VIOLATED</span>
              </>
            )}
          </div>
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            Agent: <strong className="text-white">{simulation.agent_name}</strong>
          </span>
        </div>

        {/* Live KPI telemetry */}
        <div className="flex items-center space-x-4 text-xs font-mono">
          <div className="flex items-center space-x-1.5">
            <DollarSign className="h-4 w-4 text-emerald-400" />
            <span className="text-slate-400">Total Cost:</span>
            <span className="text-emerald-400 font-bold">
              ${currentStep.cumulative_cost.toLocaleString()}
            </span>
          </div>
          <div className="flex items-center space-x-1.5">
            <Clock className="h-4 w-4 text-amber-400" />
            <span className="text-slate-400">Days Elapsed:</span>
            <span className="text-amber-400 font-bold">
              {currentStep.days_elapsed.toFixed(1)} / {simulation.initial_deadline.toFixed(1)}d
            </span>
          </div>
          <div className="flex items-center space-x-1.5">
            <Package className="h-4 w-4 text-cyan-400" />
            <span className="text-slate-400">Cargo:</span>
            <span className="text-cyan-400 font-bold">
              {simulation.scenario.shipment_weight}t
            </span>
          </div>
        </div>
      </div>

      {/* Disruption & Pivot Notification Alerts */}
      <div className="px-4 pt-3 space-y-2">
        {disruptionTriggeredNow && (
          <div className="p-2.5 rounded-xl bg-red-950/60 border border-red-500/60 text-red-200 text-xs flex items-center justify-between shadow-lg shadow-red-950/40 animate-pulse">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
              <span>
                <strong>CRITICAL MARITIME DISRUPTION:</strong> Fat-tailed delay shock activated (+
                {currentStep.days_passed.toFixed(1)} days) due to severe port bottleneck!
              </span>
            </div>
            <span className="px-2 py-0.5 rounded bg-red-500/20 text-red-300 font-mono text-[10px] font-bold">
              S_disrupt = {(simulation.scenario.disruption_shock_prob * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {currentStep.is_pivot && (
          <div className="p-2.5 rounded-xl bg-amber-950/70 border border-amber-500/70 text-amber-200 text-xs flex items-center justify-between shadow-lg shadow-amber-950/40">
            <div className="flex items-center space-x-2">
              <Zap className="h-4 w-4 text-amber-400 shrink-0 animate-bounce" />
              <span>
                <strong>DYNAMIC INTERMODAL PIVOT (`SWITCH_TO_AIR`):</strong> Agent detected deadline shortfall at Regional Checkpoint and aborted maritime leg to expedite cargo by Air!
              </span>
            </div>
            <span className="px-2 py-0.5 rounded bg-amber-500/30 text-amber-200 font-mono text-[10px] font-bold">
              +1.0d Handling | $2,000/t
            </span>
          </div>
        )}
      </div>

      {/* Interactive SVG Multi-Echelon Diagram */}
      <div className="relative flex-1 min-h-[380px] p-4 flex items-center justify-center">
        <svg
          viewBox="0 0 900 420"
          className="w-full h-full max-h-[480px]"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Background Grid Pattern */}
          <defs>
            <pattern
              id="grid-pattern"
              width="40"
              height="40"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 40 0 L 0 0 0 40"
                fill="none"
                stroke="rgba(51, 65, 85, 0.2)"
                strokeWidth="1"
              />
            </pattern>

            {/* Linear & Radial Gradients */}
            <linearGradient id="airGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#a855f7" />
            </linearGradient>

            <linearGradient id="oceanGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0284c7" />
              <stop offset="100%" stopColor="#0d9488" />
            </linearGradient>

            <linearGradient id="pivotGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>

            <filter id="glowAir" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            <filter id="glowPivot" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          <rect width="900" height="420" fill="url(#grid-pattern)" />

          {/* ---------------- PATH 0: DIRECT AIR LEG (Dubai -> Disaster Site) ---------------- */}
          {/* Arched high-altitude flight trajectory */}
          <path
            d="M 120 220 Q 450 60 780 220"
            stroke={directAirActive ? "url(#airGrad)" : "rgba(71, 85, 105, 0.4)"}
            strokeWidth={directAirActive ? "4" : "2"}
            strokeDasharray={directAirActive ? "8 4" : "4 4"}
            fill="none"
            filter={directAirActive ? "url(#glowAir)" : undefined}
            className={directAirActive ? "animate-pulse" : ""}
          />
          <text
            x="450"
            y="95"
            textAnchor="middle"
            fill={directAirActive ? "#38bdf8" : "#64748b"}
            className="text-[11px] font-mono font-semibold"
          >
            Direct Air Corridor ($2,000/t • 2.0d)
          </text>

          {/* ---------------- PATH 1: LEG 1 OCEAN (Dubai -> Regional Checkpoint) ---------------- */}
          <path
            d="M 120 220 C 220 300, 320 300, 450 240"
            stroke={
              oceanLeg1Active
                ? disruptionTriggeredNow && activeAction === 1
                  ? "#ef4444"
                  : "#0284c7"
                : "rgba(71, 85, 105, 0.3)"
            }
            strokeWidth={oceanLeg1Active ? "4" : "2"}
            fill="none"
            strokeDasharray={oceanLeg1Active ? "none" : "3 3"}
          />
          <text
            x="270"
            y="300"
            textAnchor="middle"
            fill={oceanLeg1Active ? "#38bdf8" : "#64748b"}
            className="text-[11px] font-mono"
          >
            Ocean Leg 1 ($200/t • 7.0d base)
          </text>

          {/* ---------------- PATH 2: LEG 1 LAND (Dubai -> Regional Checkpoint) ---------------- */}
          <path
            d="M 120 220 L 450 240"
            stroke={landLeg1Active ? "#10b981" : "rgba(71, 85, 105, 0.25)"}
            strokeWidth={landLeg1Active ? "4" : "1.5"}
            strokeDasharray="6 4"
            fill="none"
          />
          <text
            x="280"
            y="215"
            textAnchor="middle"
            fill={landLeg1Active ? "#34d399" : "#64748b"}
            className="text-[10px] font-mono"
          >
            Overland Leg 1 ($500/t • 4.0d)
          </text>

          {/* ---------------- PATH 3: LEG 2 NOMINAL (Regional Checkpoint -> Disaster Site) ---------------- */}
          <path
            d="M 450 240 C 560 300, 680 300, 780 220"
            stroke={
              continueNominalActive
                ? disruptionTriggeredNow && activeAction === 4
                  ? "#ef4444"
                  : "#0ea5e9"
                : "rgba(71, 85, 105, 0.3)"
            }
            strokeWidth={continueNominalActive ? "4" : "2"}
            fill="none"
            strokeDasharray={continueNominalActive ? "none" : "3 3"}
          />
          <text
            x="620"
            y="300"
            textAnchor="middle"
            fill={continueNominalActive ? "#38bdf8" : "#64748b"}
            className="text-[11px] font-mono"
          >
            Nominal Leg 2 ($100/t • 5.0d)
          </text>

          {/* ---------------- PATH 4: EMERGENCY PIVOT TO AIR (Regional Checkpoint -> Disaster Site) ---------------- */}
          <path
            d="M 450 240 Q 610 130 780 220"
            stroke={pivotAirActive ? "url(#pivotGrad)" : "rgba(245, 158, 11, 0.25)"}
            strokeWidth={pivotAirActive ? "5" : "2"}
            strokeDasharray={pivotAirActive ? "8 4" : "4 4"}
            fill="none"
            filter={pivotAirActive ? "url(#glowPivot)" : undefined}
            className={pivotAirActive ? "animate-pulse" : ""}
          />
          <text
            x="620"
            y="155"
            textAnchor="middle"
            fill={pivotAirActive ? "#fbbf24" : "#78716c"}
            className="text-[11px] font-mono font-bold"
          >
            ⚡ Emergency Pivot: Air Cargo ($2,000/t + $500 switch)
          </text>

          {/* ---------------- MARITIME BOTTLENECK HAZARD BADGE ---------------- */}
          {anyDisruption && (
            <g transform="translate(260, 275)">
              <rect
                x="-15"
                y="-15"
                width="30"
                height="30"
                rx="6"
                fill="#7f1d1d"
                stroke="#ef4444"
                strokeWidth="2"
                className="animate-hazard"
              />
              <text
                x="0"
                y="5"
                textAnchor="middle"
                fill="#fca5a5"
                fontSize="14"
                fontWeight="bold"
              >
                ⚠️
              </text>
            </g>
          )}

          {/* ---------------- NODE 1: DUBAI ORIGIN ---------------- */}
          <g transform="translate(120, 220)">
            <circle
              r="40"
              fill="#0f172a"
              stroke="#10b981"
              strokeWidth="3"
              className="drop-shadow-lg"
            />
            <circle
              r="46"
              fill="none"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeDasharray="4 4"
              className="animate-spin"
              style={{ transformOrigin: "0 0", animationDuration: "12s" }}
            />
            <foreignObject x="-24" y="-24" width="48" height="48">
              <div className="flex items-center justify-center w-full h-full text-emerald-400">
                <Plane className="h-6 w-6" />
              </div>
            </foreignObject>
            <text
              y="60"
              textAnchor="middle"
              fill="#f8fafc"
              className="text-xs font-bold font-sans"
            >
              DUBAI ORIGIN
            </text>
            <text
              y="74"
              textAnchor="middle"
              fill="#94a3b8"
              className="text-[10px] font-mono"
            >
              International Humanitarian City
            </text>
          </g>

          {/* ---------------- NODE 2: REGIONAL CHECKPOINT ---------------- */}
          <g transform="translate(450, 240)">
            <circle
              r="42"
              fill="#0f172a"
              stroke={pivotAirActive ? "#f59e0b" : "#3b82f6"}
              strokeWidth="3"
              className="drop-shadow-lg"
            />
            <circle
              r="48"
              fill="none"
              stroke={pivotAirActive ? "#f59e0b" : "#3b82f6"}
              strokeWidth="1.5"
              strokeDasharray="4 4"
              className="animate-spin"
              style={{ transformOrigin: "0 0", animationDuration: "16s" }}
            />
            <foreignObject x="-24" y="-24" width="48" height="48">
              <div className="flex items-center justify-center w-full h-full text-cyan-400">
                {pivotAirActive ? (
                  <Zap className="h-6 w-6 text-amber-400 animate-pulse" />
                ) : (
                  <Ship className="h-6 w-6 text-blue-400" />
                )}
              </div>
            </foreignObject>
            <text
              y="62"
              textAnchor="middle"
              fill="#f8fafc"
              className="text-xs font-bold font-sans"
            >
              REGIONAL CHECKPOINT
            </text>
            <text
              y="76"
              textAnchor="middle"
              fill="#94a3b8"
              className="text-[10px] font-mono"
            >
              Intermodal Staging Hub
            </text>
          </g>

          {/* ---------------- NODE 3: DISASTER SITE ---------------- */}
          <g transform="translate(780, 220)">
            <circle
              r="40"
              fill="#0f172a"
              stroke={isFinalStep && !isDelivered ? "#ef4444" : "#10b981"}
              strokeWidth="3"
              className="drop-shadow-lg"
            />
            <circle
              r="46"
              fill="none"
              stroke={isFinalStep && !isDelivered ? "#ef4444" : "#10b981"}
              strokeWidth="1.5"
              strokeDasharray="4 4"
              className="animate-spin"
              style={{ transformOrigin: "0 0", animationDuration: "10s" }}
            />
            <foreignObject x="-24" y="-24" width="48" height="48">
              <div className="flex items-center justify-center w-full h-full text-emerald-400">
                <MapPin
                  className={`h-6 w-6 ${
                    isFinalStep && !isDelivered ? "text-red-400" : "text-emerald-400"
                  }`}
                />
              </div>
            </foreignObject>
            <text
              y="60"
              textAnchor="middle"
              fill="#f8fafc"
              className="text-xs font-bold font-sans"
            >
              DISASTER SITE
            </text>
            <text
              y="74"
              textAnchor="middle"
              fill="#94a3b8"
              className="text-[10px] font-mono"
            >
              Forward Crisis Operations
            </text>
          </g>

          {/* Animated Cargo Packet indicator based on current step */}
          {activeAction === 0 && (
            <g className="animate-pulse">
              <circle cx="450" cy="140" r="12" fill="#38bdf8" />
              <text x="450" y="144" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="bold">
                ✈️
              </text>
            </g>
          )}

          {activeAction === 1 && (
            <g className="animate-pulse">
              <circle cx="280" cy="275" r="12" fill="#0284c7" />
              <text x="280" y="279" textAnchor="middle" fill="#ffffff" fontSize="11" fontWeight="bold">
                🚢
              </text>
            </g>
          )}

          {activeAction === 2 && (
            <g className="animate-pulse">
              <circle cx="280" cy="230" r="12" fill="#10b981" />
              <text x="280" y="234" textAnchor="middle" fill="#ffffff" fontSize="11" fontWeight="bold">
                🚚
              </text>
            </g>
          )}

          {activeAction === 3 && (
            <g className="animate-pulse">
              <circle cx="620" cy="185" r="14" fill="#f59e0b" filter="url(#glowPivot)" />
              <text x="620" y="189" textAnchor="middle" fill="#0f172a" fontSize="12" fontWeight="bold">
                ⚡✈️
              </text>
            </g>
          )}

          {activeAction === 4 && (
            <g className="animate-pulse">
              <circle cx="620" cy="285" r="12" fill="#0ea5e9" />
              <text x="620" y="289" textAnchor="middle" fill="#ffffff" fontSize="11" fontWeight="bold">
                🚢
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Bottom Step Notes Footer */}
      <div className="p-3 bg-slate-900/60 border-t border-slate-800 text-xs text-slate-300 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-white">Current Echelon:</span>
          <span className="px-2 py-0.5 rounded bg-slate-800 font-mono text-emerald-400">
            {currentStep.location_start_name} ➔ {currentStep.location_end_name}
          </span>
          <span className="text-slate-400 hidden md:inline">
            | Action: <span className="text-cyan-300">{currentStep.action_name}</span> ({currentStep.transport_mode})
          </span>
        </div>
        <div className="text-[11px] font-mono text-slate-400">
          Slack:{" "}
          <span
            className={
              currentStep.deadline_remaining >= 0 ? "text-emerald-400 font-bold" : "text-red-400 font-bold"
            }
          >
            {currentStep.deadline_remaining.toFixed(1)} days
          </span>
        </div>
      </div>
    </div>
  );
};
