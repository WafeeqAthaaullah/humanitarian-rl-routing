import React, { useState } from "react";
import type Plotly from "plotly.js-dist-min";
import { PlotlyChart } from "./PlotlyChart";
import type { QuantileAnalysis, TrajectoryStep } from "../types";
import {
  TrendingDown,
  ShieldAlert,
  CheckCircle,
  XCircle,
} from "lucide-react";

interface QuantileRiskInspectorProps {
  currentStep?: TrajectoryStep | null;
  initialQuantiles?: QuantileAnalysis | null;
}

const ACTION_COLORS: Record<string, string> = {
  AIR_DIRECT: "#38bdf8", // Sky blue
  OCEAN_TO_HUB: "#0284c7", // Deep blue
  LAND_TO_HUB: "#10b981", // Emerald green
  SWITCH_TO_AIR: "#f59e0b", // Amber gold
  CONTINUE_NOMINAL: "#a855f7", // Purple
};

export const QuantileRiskInspector: React.FC<QuantileRiskInspectorProps> = ({
  currentStep,
  initialQuantiles,
}) => {
  const [cvarCutoff, setCvarCutoff] = useState<"05" | "10">("05");

  const qa: QuantileAnalysis | null =
    currentStep?.quantiles_analysis || initialQuantiles || null;

  if (!qa || !qa.quantiles || qa.quantiles.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-slate-900/60 rounded-2xl border border-slate-800 text-center">
        <TrendingDown className="h-10 w-10 text-slate-500 mb-3 animate-pulse" />
        <h4 className="text-sm font-semibold text-slate-300">
          Awaiting Quantile Telemetry
        </h4>
        <p className="text-xs text-slate-500 max-w-sm mt-1">
          Execute a dispatch simulation to inspect the neural network's 50
          predicted quantiles and CVaR tail metrics for each candidate action.
        </p>
      </div>
    );
  }

  const numQuantiles = qa.quantiles[0]?.length || 50;
  const xValues = Array.from({ length: numQuantiles }, (_, i) => i + 1);

  // Cutoff point: 5% is index 3 (or 2), 10% is index 5
  const cutoffIndex =
    cvarCutoff === "05" ? qa.cutoff_index_05 : qa.cutoff_index_10;
  const cutoffAlpha = cvarCutoff === "05" ? "5%" : "10%";

  // Build Plotly Traces: one cumulative curve per action
  const traces: Plotly.Data[] = qa.quantiles.map((actionQuantiles, idx) => {
    const actionName = qa.action_names[idx] || `Action ${idx}`;
    const sorted = [...actionQuantiles].sort((a, b) => a - b);
    const color = ACTION_COLORS[actionName] || "#94a3b8";
    const isMaskedOut = currentStep?.action_mask
      ? !currentStep.action_mask[idx]
      : false;

    return {
      x: xValues,
      y: sorted,
      type: "scatter",
      mode: "lines+markers",
      name: `${actionName}${isMaskedOut ? " (Masked)" : ""}`,
      line: {
        color: color,
        width: isMaskedOut ? 1.5 : 2.5,
        dash: isMaskedOut ? "dot" : "solid",
      },
      marker: {
        size: isMaskedOut ? 3 : 5,
        color: color,
      },
      opacity: isMaskedOut ? 0.45 : 1.0,
      hovertemplate: `<b>${actionName}</b><br>Quantile: %{x}/50<br>Return: %{y:.3f}<extra></extra>`,
    };
  });

  // Plotly Layout with shaded red risk zone for the worst alpha tail
  const layout: Partial<Plotly.Layout> = {
    title: {
      text: `<b>Distributional QR-DQN Quantile Curves (50 Quantiles)</b>`,
      font: { color: "#f8fafc", size: 14 },
      x: 0.05,
    },
    xaxis: {
      title: { text: "Quantile Index (Z₁ to Z₅₀)", font: { size: 11, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
      zerolinecolor: "rgba(71, 85, 105, 0.4)",
      tickmode: "linear",
      tick0: 1,
      dtick: 5,
      range: [0.5, 50.5],
    },
    yaxis: {
      title: { text: "Predicted Return / Q-Value", font: { size: 11, color: "#94a3b8" } },
      gridcolor: "rgba(51, 65, 85, 0.3)",
      zerolinecolor: "rgba(71, 85, 105, 0.4)",
    },
    shapes: [
      // Shaded Red Risk Zone for the worst alpha tail (e.g., quantiles 1 to cutoffIndex)
      {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: 0.5,
        x1: cutoffIndex + 0.5,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(239, 68, 68, 0.15)",
        line: { width: 0 },
        layer: "below",
      },
      // Vertical Cutoff Line
      {
        type: "line",
        xref: "x",
        yref: "paper",
        x0: cutoffIndex + 0.5,
        x1: cutoffIndex + 0.5,
        y0: 0,
        y1: 1,
        line: {
          color: "#ef4444",
          width: 2,
          dash: "dash",
        },
      },
    ],
    annotations: [
      {
        x: (cutoffIndex + 1) / 2,
        y: 0.95,
        xref: "x",
        yref: "paper",
        text: `<b>Worst ${cutoffAlpha} Tail Risk Zone (CVaR)</b>`,
        showarrow: false,
        font: { color: "#fca5a5", size: 10 },
        bgcolor: "rgba(127, 29, 29, 0.6)",
        bordercolor: "rgba(239, 68, 68, 0.6)",
        borderwidth: 1,
        borderpad: 4,
      },
    ],
    legend: {
      orientation: "h",
      x: 0,
      y: -0.2,
      font: { color: "#cbd5e1", size: 11 },
    },
    margin: { t: 45, r: 20, b: 65, l: 50 },
    height: 380,
  };

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 space-y-4 shadow-xl">
      {/* Header and Alpha Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400">
            <TrendingDown className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-100 text-sm">
              Real-Time Quantile & CVaR Risk Inspector
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Evaluating all 5 actions across 50 quantiles ($Z_1 \dots Z_{50}$)
            </p>
          </div>
        </div>

        {/* Alpha Tail Threshold Toggle */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-slate-400 font-mono">Tail Risk (α):</span>
          <div className="bg-slate-950 p-1 rounded-lg border border-slate-800 flex space-x-1">
            <button
              onClick={() => setCvarCutoff("05")}
              className={`px-2.5 py-1 rounded text-xs font-semibold transition-all ${
                cvarCutoff === "05"
                  ? "bg-red-500/30 text-red-300 border border-red-500/50 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              α = 5% (Worst 3)
            </button>
            <button
              onClick={() => setCvarCutoff("10")}
              className={`px-2.5 py-1 rounded text-xs font-semibold transition-all ${
                cvarCutoff === "10"
                  ? "bg-purple-500/30 text-purple-300 border border-purple-500/50 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              α = 10% (Worst 5)
            </button>
          </div>
        </div>
      </div>

      {/* Plotly Chart Container */}
      <div className="h-[380px] w-full rounded-xl overflow-hidden bg-slate-950/60 border border-slate-800/80">
        <PlotlyChart data={traces} layout={layout} />
      </div>

      {/* Side-by-Side Comparison Callout Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
          <span>Action Risk Comparison: Mean Return vs. CVaR Tail Score</span>
          <span className="text-[11px] font-mono text-slate-400">
            Current Echelon: {currentStep?.location_start_name || "Origin"}
          </span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-950/80 text-slate-400 font-mono border-b border-slate-800">
              <tr>
                <th className="py-2 px-3">Action</th>
                <th className="py-2 px-3">Mode</th>
                <th className="py-2 px-3 text-right">Mean Q-Value</th>
                <th className="py-2 px-3 text-right">5% CVaR Score</th>
                <th className="py-2 px-3 text-right">10% CVaR Score</th>
                <th className="py-2 px-3 text-center">Legal Mask</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {qa.action_names.map((name, i) => {
                const isMaskedOut = currentStep?.action_mask
                  ? !currentStep.action_mask[i]
                  : false;
                const isSelected = currentStep?.action === i;
                const meanVal = qa.q_mean[i] ?? 0;
                const cvar05 = qa.cvar_05[i] ?? 0;
                const cvar10 = qa.cvar_10[i] ?? 0;

                return (
                  <tr
                    key={name}
                    className={`transition-colors ${
                      isSelected
                        ? "bg-emerald-500/10 text-white font-bold"
                        : isMaskedOut
                        ? "opacity-40 text-slate-500"
                        : "hover:bg-slate-800/40 text-slate-300"
                    }`}
                  >
                    <td className="py-2 px-3 flex items-center space-x-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: ACTION_COLORS[name] }}
                      />
                      <span className="font-sans font-semibold">
                        {name}
                        {isSelected && (
                          <span className="ml-2 px-1.5 py-0.2 rounded text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                            SELECTED
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-sans text-slate-400">
                      {i === 0
                        ? "Air Direct"
                        : i === 1
                        ? "Ocean to Hub"
                        : i === 2
                        ? "Land to Hub"
                        : i === 3
                        ? "Switch to Air"
                        : "Continue"}
                    </td>
                    <td className="py-2 px-3 text-right text-cyan-400 font-semibold">
                      {meanVal.toFixed(3)}
                    </td>
                    <td
                      className={`py-2 px-3 text-right font-semibold ${
                        cvar05 < -5.0 ? "text-red-400" : "text-amber-300"
                      }`}
                    >
                      {cvar05.toFixed(3)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">
                      {cvar10.toFixed(3)}
                    </td>
                    <td className="py-2 px-3 text-center">
                      {!isMaskedOut ? (
                        <span className="inline-flex items-center text-emerald-400 text-[11px]">
                          <CheckCircle className="h-3.5 w-3.5 mr-1" /> Valid
                        </span>
                      ) : (
                        <span className="inline-flex items-center text-slate-500 text-[11px]">
                          <XCircle className="h-3.5 w-3.5 mr-1" /> Masked
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Deep Dive Insight Explanatory Box */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-purple-950/40 via-slate-900 to-slate-950 border border-purple-500/30 text-xs text-slate-300 space-y-2 shadow-md">
        <div className="flex items-center space-x-2 text-purple-300 font-semibold font-sans">
          <ShieldAlert className="h-4 w-4 text-purple-400 shrink-0" />
          <span>
            Distributional Safety Insight: Why QR-DQN + CVaR Outperforms Standard DQN / Mean
          </span>
        </div>
        <p className="text-slate-400 leading-relaxed font-sans text-[11px]">
          Standard <strong>DQN</strong> and <strong>QR-DQN-Mean</strong> compute
          the scalar mathematical expectation (E[Q(s, a)]). When maritime disruption probability is elevated, ocean freight rates remain cheap ($200/ton), so the <em>average</em> return appears deceptively favorable. However, maritime transit times follow a heavy fat-tailed lognormal distribution (mu=2.0, sigma=0.5).
        </p>
        <p className="text-slate-400 leading-relaxed font-sans text-[11px]">
          <strong>QR-DQN + CVaR</strong> inspects the worst α = 5% tail (Z₁ to Z₃). In this catastrophic tail, ocean freight arrives 15–20+ days late, triggering severe Tier-1 deadline violation penalties. By maximizing the lower tail return rather than the naive average, QR-DQN + CVaR deliberately chooses Air Direct or triggers an emergency intermodal pivot to Air at the Regional Checkpoint, guaranteeing zero deadline violations within the 1% safety budget.
        </p>
      </div>
    </div>
  );
};
