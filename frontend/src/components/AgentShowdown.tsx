import React from "react";
import {
  GitCompare,
  Zap,
  CheckCircle2,
  XCircle,
  Play,
  RotateCw,
  Trophy,
} from "lucide-react";
import type { ShowdownResponse, ModelShowdownResult } from "../types";

interface AgentShowdownProps {
  showdownData: ShowdownResponse | null;
  loading: boolean;
  onRunShowdown: () => void;
  onSelectAgentForSim: (agentId: string) => void;
}

export const AgentShowdown: React.FC<AgentShowdownProps> = ({
  showdownData,
  loading,
  onRunShowdown,
  onSelectAgentForSim,
}) => {
  if (!showdownData && !loading) {
    return (
      <div className="flex flex-col items-center justify-center p-12 bg-slate-900/60 rounded-2xl border border-slate-800 text-center">
        <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-4 animate-pulse">
          <GitCompare className="h-8 w-8" />
        </div>
        <h3 className="text-lg font-semibold text-slate-200">
          7-Model Showdown Benchmark
        </h3>
        <p className="text-xs text-slate-400 max-w-lg mt-2">
          Run the exact same seeded disruption scenario simultaneously across all
          7 models: Greedy, Static Threshold, Random, DQN, DDQN, QR-DQN-Mean, and
          QR-DQN + CVaR to compare safety, freight cost, and intermodal pivot recovery.
        </p>
        <button
          onClick={onRunShowdown}
          className="mt-6 px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-semibold shadow-lg shadow-cyan-950/60 flex items-center space-x-2"
        >
          <Zap className="h-4 w-4" />
          <span>Launch 7-Model Showdown</span>
        </button>
      </div>
    );
  }

  const results: ModelShowdownResult[] = showdownData?.results || [];

  const deliveredCount = results.filter((r) => r.status === "Delivered").length;
  const violatedCount = results.filter((r) => r.status !== "Delivered").length;
  const pivotedCount = results.filter((r) => r.pivoted).length;

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 space-y-6 shadow-xl">
      {/* Header and Re-run Button */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <GitCompare className="h-6 w-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-base font-bold text-white tracking-tight">
                7-Model Synchronous Showdown
              </h2>
              <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                Seeded Multi-Agent Run
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Identical Scenario: Urgency={showdownData?.scenario.urgency_tier === 1 ? "Tier 1 (Critical)" : "Tier 2"} • Weight={showdownData?.scenario.shipment_weight}t • Deadline={showdownData?.scenario.deadline_remaining}d • Disruption={((showdownData?.scenario.disruption_shock_prob ?? 0) * 100).toFixed(0)}% • Seed={showdownData?.seed}
            </p>
          </div>
        </div>

        <button
          onClick={onRunShowdown}
          disabled={loading}
          className={`px-4 py-2 rounded-xl text-xs font-semibold text-white transition-all flex items-center space-x-2 shadow-md ${
            loading
              ? "bg-slate-800 text-slate-500 cursor-not-allowed"
              : "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 active:scale-95 shadow-cyan-950/50"
          }`}
        >
          <RotateCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>{loading ? "Evaluating Models..." : "Re-run Showdown"}</span>
        </button>
      </div>

      {/* Summary Stat Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
          <span className="text-[11px] text-slate-400 flex items-center space-x-1">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            <span>Delivered On-Time</span>
          </span>
          <div className="text-xl font-bold font-mono text-emerald-400">
            {deliveredCount} / {results.length}
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
          <span className="text-[11px] text-slate-400 flex items-center space-x-1">
            <XCircle className="h-3.5 w-3.5 text-red-400" />
            <span>Violated Deadline</span>
          </span>
          <div className="text-xl font-bold font-mono text-red-400">
            {violatedCount} / {results.length}
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
          <span className="text-[11px] text-slate-400 flex items-center space-x-1">
            <Zap className="h-3.5 w-3.5 text-amber-400" />
            <span>Intermodal Pivots</span>
          </span>
          <div className="text-xl font-bold font-mono text-amber-400">
            {pivotedCount} Models
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
          <span className="text-[11px] text-slate-400 flex items-center space-x-1">
            <Trophy className="h-3.5 w-3.5 text-cyan-400" />
            <span>Champion Model</span>
          </span>
          <div className="text-xs font-bold text-white truncate pt-1">
            QR-DQN + CVaR
          </div>
        </div>
      </div>

      {/* Comprehensive Side-by-Side Comparison Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/40">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-950 text-slate-400 font-mono border-b border-slate-800">
            <tr>
              <th className="py-3 px-4">Model & Architecture</th>
              <th className="py-3 px-3">Family</th>
              <th className="py-3 px-3 text-center">Status</th>
              <th className="py-3 px-3 text-right">Freight Cost ($)</th>
              <th className="py-3 px-3 text-right">Elapsed Days</th>
              <th className="py-3 px-3 text-center">Intermodal Pivot</th>
              <th className="py-3 px-4">Action Sequence</th>
              <th className="py-3 px-3 text-center">Simulator</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {results.map((res) => {
              const isDelivered = res.status === "Delivered";
              const isChampion = res.agent_id === "qr_dqn_cvar";

              return (
                <tr
                  key={res.agent_id}
                  className={`transition-colors ${
                    isChampion
                      ? "bg-emerald-500/5 hover:bg-emerald-500/10"
                      : "hover:bg-slate-800/30"
                  }`}
                >
                  {/* Model Name */}
                  <td className="py-3 px-4 font-semibold text-slate-100 flex items-center space-x-2">
                    {isChampion && (
                      <span title="Champion Model">
                        <Trophy className="h-4 w-4 text-emerald-400 shrink-0" />
                      </span>
                    )}
                    <span>{res.agent_name}</span>
                  </td>

                  {/* Architecture Family */}
                  <td className="py-3 px-3 text-slate-400 font-mono text-[11px]">
                    {res.family}
                  </td>

                  {/* Delivery Status Badge */}
                  <td className="py-3 px-3 text-center">
                    <span
                      className={`inline-flex items-center space-x-1 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border ${
                        isDelivered
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                          : "bg-red-500/20 text-red-300 border-red-500/40"
                      }`}
                    >
                      {isDelivered ? (
                        <>
                          <CheckCircle2 className="h-3 w-3 mr-0.5 text-emerald-400" />
                          <span>DELIVERED</span>
                        </>
                      ) : (
                        <>
                          <XCircle className="h-3 w-3 mr-0.5 text-red-400" />
                          <span>DEADLINE VIOLATED</span>
                        </>
                      )}
                    </span>
                  </td>

                  {/* Freight Cost */}
                  <td className="py-3 px-3 text-right font-mono font-semibold text-emerald-400">
                    ${res.total_freight_cost.toLocaleString()}
                  </td>

                  {/* Transit Days vs Deadline */}
                  <td className="py-3 px-3 text-right font-mono text-slate-200">
                    <span
                      className={
                        res.deadline_remaining >= 0
                          ? "text-slate-200"
                          : "text-red-400 font-bold"
                      }
                    >
                      {res.total_days.toFixed(1)}d
                    </span>
                    <span className="text-slate-500 text-[10px] block">
                      (slack: {res.deadline_remaining.toFixed(1)}d)
                    </span>
                  </td>

                  {/* Intermodal Pivot */}
                  <td className="py-3 px-3 text-center">
                    {res.pivoted ? (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                        <Zap className="h-3 w-3 mr-1 text-amber-400" />
                        SWITCH_TO_AIR
                      </span>
                    ) : (
                      <span className="text-slate-500 font-mono text-[11px]">
                        Nominal
                      </span>
                    )}
                  </td>

                  {/* Route Sequence */}
                  <td className="py-3 px-4 font-mono text-[11px] text-slate-300">
                    <div className="flex items-center space-x-1.5 flex-wrap">
                      {res.action_names.map((actName, idx) => (
                        <React.Fragment key={idx}>
                          <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/60 text-slate-200 text-[10px]">
                            {actName}
                          </span>
                          {idx < res.action_names.length - 1 && (
                            <span className="text-slate-600">➔</span>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  </td>

                  {/* Inspect in Simulator Action */}
                  <td className="py-3 px-3 text-center">
                    <button
                      onClick={() => onSelectAgentForSim(res.agent_id)}
                      className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-all text-[11px] flex items-center space-x-1 mx-auto"
                      title="Load this agent into the Simulator timeline"
                    >
                      <Play className="h-3 w-3 text-emerald-400" />
                      <span>Inspect</span>
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
