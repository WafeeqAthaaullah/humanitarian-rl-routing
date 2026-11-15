import React from "react";
import {
  Compass,
  GitCompare,
  TrendingDown,
  Activity,
  Layers,
  ShieldCheck,
} from "lucide-react";

interface NavbarProps {
  activeTab: "simulator" | "showdown" | "quantiles" | "training";
  setActiveTab: (tab: "simulator" | "showdown" | "quantiles" | "training") => void;
  serverOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  serverOnline,
}) => {
  return (
    <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Project Title */}
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shadow-lg shadow-emerald-950/50">
              <Compass className="h-6 w-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold tracking-tight text-white text-lg sm:text-xl">
                  DUBAI HUMANITARIAN RL
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold uppercase tracking-wider rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  Command Center
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono hidden sm:block">
                3-Echelon Intermodal Routing • Risk-Averse QR-DQN with CVaR
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => setActiveTab("simulator")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === "simulator"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent"
              }`}
            >
              <Layers className="h-4 w-4" />
              <span>Simulator</span>
            </button>

            <button
              onClick={() => setActiveTab("showdown")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === "showdown"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent"
              }`}
            >
              <GitCompare className="h-4 w-4" />
              <span>7-Model Showdown</span>
            </button>

            <button
              onClick={() => setActiveTab("quantiles")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === "quantiles"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent"
              }`}
            >
              <TrendingDown className="h-4 w-4" />
              <span>CVaR Inspector</span>
            </button>

            <button
              onClick={() => setActiveTab("training")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === "training"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent"
              }`}
            >
              <Activity className="h-4 w-4" />
              <span>Diagnostics</span>
            </button>
          </nav>

          {/* System Status Indicator */}
          <div className="hidden lg:flex items-center space-x-3 text-xs font-mono">
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-slate-300">CMDP Safety Budget: 1%</span>
            </div>
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">
              <span
                className={`h-2 w-2 rounded-full ${
                  serverOnline ? "bg-emerald-400 animate-ping" : "bg-red-400"
                }`}
              />
              <span className={serverOnline ? "text-emerald-400" : "text-red-400"}>
                {serverOnline ? "FASTAPI ONLINE" : "OFFLINE"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
