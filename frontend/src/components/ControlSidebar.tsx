import React, { useState } from "react";
import { AlertTriangle, ChevronDown, ChevronUp, Dice5, Layers, Pause, Play, RotateCcw, Send, SkipBack, SkipForward, Sliders } from "lucide-react";
import type { SimulationRequest, TrajectoryStep } from "../types";
import { CHECKPOINT_HUBS, DESTINATION_ZONES, ORIGIN_HUBS, type InternationalRoute, type InternationalTransportMode } from "../config/internationalRoutes";

interface ControlSidebarProps {
  request: SimulationRequest;
  setRequest: React.Dispatch<React.SetStateAction<SimulationRequest>>;
  route: InternationalRoute;
  setRoute: React.Dispatch<React.SetStateAction<InternationalRoute>>;
  onSimulate: () => void;
  loading: boolean;
  trajectory: TrajectoryStep[];
  currentStepIndex: number;
  setCurrentStepIndex: (idx: number) => void;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;
}

export const AGENT_OPTIONS = [
  ["qr_dqn_cvar", "QR-DQN CVaR", "Champion · distributional tail-risk policy"],
  ["qr_dqn_mean", "QR-DQN Mean", "Risk-neutral distributional baseline"],
  ["ddqn", "DDQN-Lagrangian", "Constrained policy optimization"],
  ["dqn", "DQN", "Standard value-learning baseline"],
  ["static_threshold", "Static Threshold", "Domain heuristic baseline"],
  ["greedy", "Greedy Cheapest", "Cost-first heuristic"],
  ["random", "Random Masked", "Legal-action sanity baseline"],
] as const;

const modes: Array<[InternationalTransportMode, string]> = [
  ["ocean", "Ocean Freight"],
  ["air", "Air Cargo"],
  ["overland", "Overland Convoy"],
  ["multimodal", "Multimodal / RL Auto"],
];

export const ControlSidebar: React.FC<ControlSidebarProps> = ({
  request, setRequest, route, setRoute, onSimulate, loading, trajectory, currentStepIndex, setCurrentStepIndex, isPlaying, setIsPlaying,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const currentStep = trajectory[currentStepIndex];
  const legProgress = trajectory.length > 1 ? currentStepIndex / (trajectory.length - 1) : 0;
  const setWaypoint = (key: "origin" | "checkpoint" | "destination", id: string) => {
    const source = key === "origin" ? ORIGIN_HUBS : key === "checkpoint" ? CHECKPOINT_HUBS : DESTINATION_ZONES;
    const waypoint = source.find(item => item.id === id);
    if (waypoint) setRoute(previous => ({ ...previous, [key]: waypoint }));
  };
  const setMode = (mode: InternationalTransportMode) => setRoute(previous => ({ ...previous, mode }));
  const randomizeSeed = () => setRequest(previous => ({ ...previous, seed: Math.floor(Math.random() * 99999) + 1 }));
  const setPlaybackProgress = (value: number) => {
    if (trajectory.length) setCurrentStepIndex(Math.round(value * (trajectory.length - 1)));
  };

  return (
    <div className="sidebar-controls international-sidebar flex flex-col h-full bg-slate-900/90 border-r border-slate-800 p-4 space-y-4 w-[360px] min-w-0 overflow-y-auto overflow-x-hidden box-border">
      <header className="sidebar-heading w-full box-border">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0"><Sliders className="h-5 w-5 text-emerald-400 shrink-0" /><h2 className="font-semibold text-sm text-slate-100 break-words">International Dispatch &amp; Corridor Configurator</h2></div>
          <span className="text-[10px] font-mono px-2 py-1 rounded bg-slate-800 text-slate-400 shrink-0">GLOBAL MDP</span>
        </div>
        <p className="text-[11px] text-slate-400 mt-2">Dubai relief hubs → regional transshipment → international crisis zones</p>
      </header>

      <section className="sidebar-section space-y-2">
        <div className="sidebar-section-title"><span>01</span><strong>Origin hub · global staging</strong></div>
        <select value={route.origin.id} onChange={event => setWaypoint("origin", event.target.value)} className="sidebar-select w-full box-border">{ORIGIN_HUBS.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select>
        <small className="route-hint">{route.origin.description} · {route.origin.coordinates.join(", ")}</small>
      </section>
      <section className="sidebar-section space-y-2">
        <div className="sidebar-section-title"><span>02</span><strong>Checkpoint hub · regional customs</strong></div>
        <select value={route.checkpoint.id} onChange={event => setWaypoint("checkpoint", event.target.value)} className="sidebar-select w-full box-border">{CHECKPOINT_HUBS.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select>
        <small className="route-hint">{route.checkpoint.description} · {route.checkpoint.coordinates.join(", ")}</small>
      </section>
      <section className="sidebar-section space-y-2">
        <div className="sidebar-section-title"><span>03</span><strong>Destination zone · crisis response</strong></div>
        <select value={route.destination.id} onChange={event => setWaypoint("destination", event.target.value)} className="sidebar-select w-full box-border">{DESTINATION_ZONES.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select>
        <small className="route-hint">{route.destination.description} · {route.destination.coordinates.join(", ")}</small>
      </section>

      <section className="sidebar-section space-y-2">
        <div className="sidebar-section-title"><span>04</span><strong>Transport architecture</strong></div>
        <div className="mode-grid w-full box-border">{modes.map(([id, label]) => <button key={id} type="button" onClick={() => setMode(id)} className={`mode-option ${route.mode === id ? "active" : ""}`}>{label}</button>)}</div>
        {route.mode === "multimodal" && <div className="leg-selectors w-full box-border"><label>Leg 1 · origin → checkpoint<select value={route.leg1Mode} onChange={event => setRoute(previous => ({ ...previous, leg1Mode: event.target.value as InternationalRoute["leg1Mode"] }))}>{modes.filter(([id]) => id !== "multimodal").map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label><label>Leg 2 · checkpoint → destination<select value={route.leg2Mode} onChange={event => setRoute(previous => ({ ...previous, leg2Mode: event.target.value as InternationalRoute["leg2Mode"] }))}>{modes.filter(([id]) => id !== "multimodal").map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label></div>}
      </section>

      <section className="sidebar-section space-y-3">
        <div className="sidebar-section-title"><span>05</span><strong>Scenario &amp; RL policy</strong></div>
        <div className="scenario-field"><label>Urgency tier<select value={request.urgency_tier} onChange={event => setRequest(previous => ({ ...previous, urgency_tier: Number(event.target.value) }))}><option value={1}>Tier 1 · Critical</option><option value={2}>Tier 2 · Standard</option></select></label><label>Random seed<div className="seed-control"><input type="number" value={request.seed ?? 42} onChange={event => setRequest(previous => ({ ...previous, seed: Number(event.target.value) || 0 }))} /><button type="button" onClick={randomizeSeed} title="Randomize seed"><Dice5 size={14} /></button></div></label></div>
        <label className="range-field">Shipment weight <output>{request.shipment_weight.toFixed(1)} t</output><input type="range" min="0.5" max="50" step="0.5" value={request.shipment_weight} onChange={event => setRequest(previous => ({ ...previous, shipment_weight: Number(event.target.value) }))} /></label>
        <label className="range-field">Deadline remaining <output>{request.deadline_remaining.toFixed(1)} d</output><input type="range" min="5" max="30" step="0.5" value={request.deadline_remaining} onChange={event => setRequest(previous => ({ ...previous, deadline_remaining: Number(event.target.value) }))} /></label>
        <label className="range-field">Disruption shock probability <output>{(request.disruption_shock_prob * 100).toFixed(0)}%</output><input type="range" min="0" max="0.5" step="0.05" value={request.disruption_shock_prob} onChange={event => setRequest(previous => ({ ...previous, disruption_shock_prob: Number(event.target.value) }))} /></label>
        <div className="agent-list">{AGENT_OPTIONS.map(([id, name, description]) => <button key={id} type="button" onClick={() => setRequest(previous => ({ ...previous, agent_id: id }))} className={`agent-option ${request.agent_id === id ? "active" : ""}`}><span className="agent-option-head"><strong>{name}</strong><em>{id === "qr_dqn_cvar" ? "Champion" : id === "random" ? "Baseline" : "Policy"}</em></span><small>{description}</small></button>)}</div>
      </section>

      <section className="sidebar-section advanced-section">
        <button type="button" onClick={() => setShowAdvanced(!showAdvanced)} className="w-full box-border flex flex-wrap items-center justify-between gap-2 text-xs text-slate-300"><span>Extended state variables</span>{showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</button>
        {showAdvanced && <div className="advanced-fields"><label>Shipment volume <output>{request.volume ?? 30} m³</output><input type="range" min="5" max="100" value={request.volume ?? 30} onChange={event => setRequest(previous => ({ ...previous, volume: Number(event.target.value) }))} /></label><label>Destination inventory <output>{request.dest_inventory ?? 20} d</output><input type="range" min="0" max="50" value={request.dest_inventory ?? 20} onChange={event => setRequest(previous => ({ ...previous, dest_inventory: Number(event.target.value) }))} /></label></div>}
      </section>

      <button type="button" onClick={onSimulate} disabled={loading} className="international-dispatch-button w-full box-border">{loading ? <span className="loading-dot" /> : <Send size={16} />}<span>{loading ? "Simulating international corridor…" : "Simulate International Dispatch"}</span></button>

      <footer className="playback-footer w-full box-border">
        <div className="playback-heading w-full box-border flex flex-wrap items-center justify-between gap-2"><span><Layers size={14} /> Playback &amp; corridor progress</span><strong>Step {trajectory.length ? currentStepIndex + 1 : 0} / {trajectory.length}</strong></div>
        <input aria-label="Leg 1 and Leg 2 progress" type="range" min="0" max="1" step="0.01" value={legProgress} onChange={event => setPlaybackProgress(Number(event.target.value))} className="w-full box-border progress-slider" />
        <div className="leg-labels"><span>Leg 1 · origin → checkpoint</span><span>Leg 2 · checkpoint → destination</span></div>
        <div className="playback-buttons w-full box-border"><button type="button" onClick={() => setCurrentStepIndex(0)} title="Reset"><RotateCcw size={15} /><span>Reset</span></button><button type="button" onClick={() => setCurrentStepIndex(Math.max(0, currentStepIndex - 1))} disabled={!trajectory.length || currentStepIndex === 0} title="Step back"><SkipBack size={15} /><span>Back</span></button><button type="button" onClick={() => setIsPlaying(!isPlaying)} disabled={!trajectory.length} className="play-button"><span>{isPlaying ? <Pause size={15} /> : <Play size={15} />}</span><span>{isPlaying ? "Pause" : "Play"}</span></button><button type="button" onClick={() => setCurrentStepIndex(Math.min(trajectory.length - 1, currentStepIndex + 1))} disabled={!trajectory.length || currentStepIndex === trajectory.length - 1} title="Step forward"><SkipForward size={15} /><span>Forward</span></button></div>
        {currentStep && <div className="step-telemetry w-full box-border"><div className="step-telemetry-head"><strong>{currentStep.action_name}</strong><span>{currentStep.transport_mode}</span></div><div className="step-metrics"><span>Step Freight <b>${currentStep.freight_cost.toLocaleString()}</b></span><span>Leg Transit Time <b>+{currentStep.days_passed.toFixed(1)} d</b></span><span>Cumulative Days <b>{currentStep.days_elapsed.toFixed(1)} d</b></span><span>Deadline Slack <b>{currentStep.deadline_remaining.toFixed(1)} d</b></span></div>{currentStep.disruption_triggered && <div className="disruption-warning"><AlertTriangle size={14} /> Maritime bottleneck disruption triggered on this leg.</div>}</div>}
      </footer>
    </div>
  );
};
