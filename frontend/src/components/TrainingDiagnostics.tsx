import React, { useEffect, useState } from "react";
import type Plotly from "plotly.js-dist-min";
import { PlotlyChart } from "./PlotlyChart";
import { fetchTrainingMetrics } from "../api";
import type { TrainingMetricsResponse } from "../types";
import { Activity } from "lucide-react";

export const TrainingDiagnostics: React.FC = () => {
  const [metrics, setMetrics] = useState<TrainingMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTrainingMetrics(25)
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-12 bg-slate-900/60 rounded-2xl border border-slate-800 text-center">
        <div className="w-10 h-10 border-2 border-emerald-500/30 border-t-emerald-400 rounded-full animate-spin mb-4" />
        <p className="text-xs text-slate-400 font-mono">
          Loading 5,000-episode training metrics from logs...
        </p>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="p-6 rounded-2xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs text-center">
        Failed to load training logs: {error}
      </div>
    );
  }

  const episodes = metrics.records.map((r) => r.episode);
  const rewards = metrics.records.map((r) => r.reward);
  const rollingRewards = metrics.rolling_reward;
  const rollingViolations = metrics.rolling_violation_rate;
  const rollingCosts = metrics.rolling_freight_cost;

  // 1. Reward Convergence Plot
  const rewardTraces: Plotly.Data[] = [
    {
      x: episodes,
      y: rewards,
      type: "scatter",
      mode: "lines",
      name: "Episode Reward",
      line: { color: "rgba(56, 189, 248, 0.25)", width: 1 },
    },
    {
      x: episodes,
      y: rollingRewards,
      type: "scatter",
      mode: "lines",
      name: "100-Ep Rolling Avg",
      line: { color: "#38bdf8", width: 2.5 },
    },
  ];

  const rewardLayout: Partial<Plotly.Layout> = {
    title: {
      text: "<b>Training Reward Convergence (QR-DQN)</b>",
      font: { color: "#f8fafc", size: 13 },
      x: 0.05,
    },
    xaxis: {
      title: { text: "Episode", font: { size: 10, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
    },
    yaxis: {
      title: { text: "Normalized CMDP Reward", font: { size: 10, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
    },
    margin: { t: 40, r: 20, b: 40, l: 45 },
    height: 280,
  };

  // 2. Safety Constraint Violation Rate (%)
  const violationTraces: Plotly.Data[] = [
    {
      x: episodes,
      y: rollingViolations,
      type: "scatter",
      mode: "lines",
      name: "Tier 1 Violation Rate (%)",
      line: { color: "#ef4444", width: 2.5 },
      fill: "tozeroy",
      fillcolor: "rgba(239, 68, 68, 0.12)",
    },
    {
      x: [0, Math.max(...episodes)],
      y: [1.0, 1.0],
      type: "scatter",
      mode: "lines",
      name: "Safety Budget (1%)",
      line: { color: "#10b981", width: 2, dash: "dash" },
    },
  ];

  const violationLayout: Partial<Plotly.Layout> = {
    title: {
      text: "<b>Tier-1 Deadline Violation Rate (%) vs. Safety Budget</b>",
      font: { color: "#f8fafc", size: 13 },
      x: 0.05,
    },
    xaxis: {
      title: { text: "Episode", font: { size: 10, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
    },
    yaxis: {
      title: { text: "Violation Rate (%)", font: { size: 10, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
    },
    margin: { t: 40, r: 20, b: 40, l: 45 },
    height: 280,
  };

  // 3. Freight Cost Trajectory
  const costTraces: Plotly.Data[] = [
    {
      x: episodes,
      y: rollingCosts,
      type: "scatter",
      mode: "lines",
      name: "Rolling Avg Cost ($)",
      line: { color: "#10b981", width: 2.5 },
    },
  ];

  const costLayout: Partial<Plotly.Layout> = {
    title: {
      text: "<b>Average Freight Cost ($) Trajectory</b>",
      font: { color: "#f8fafc", size: 13 },
      x: 0.05,
    },
    xaxis: {
      title: { text: "Episode", font: { size: 10, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
    },
    yaxis: {
      title: { text: "Cost ($)", font: { size: 10, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
    },
    margin: { t: 40, r: 20, b: 40, l: 45 },
    height: 280,
  };

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 space-y-6 shadow-xl">
      {/* Title */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400">
            <Activity className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">
              Training Telemetry & Diagnostics
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Parsed from results/logs/training_metrics.csv ({metrics.total_episodes.toLocaleString()} episodes)
            </p>
          </div>
        </div>
      </div>

      {/* Grid of 3 Diagnostic Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Chart 1: Reward */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
          <PlotlyChart data={rewardTraces} layout={rewardLayout} />
        </div>

        {/* Chart 2: Constraint Violations */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
          <PlotlyChart data={violationTraces} layout={violationLayout} />
        </div>

        {/* Chart 3: Freight Cost */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 lg:col-span-2">
          <PlotlyChart data={costTraces} layout={costLayout} />
        </div>
      </div>
    </div>
  );
};
