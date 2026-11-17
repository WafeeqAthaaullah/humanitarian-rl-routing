import { useState, useEffect, useRef } from "react";
import { AlertCircle, BarChart3, CheckCircle2, CircleDollarSign, Clock3, PackageCheck, RefreshCw } from "lucide-react";
import { Navbar, type AppPage } from "./components/Navbar";
import { ControlSidebar } from "./components/ControlSidebar";
import { RouteVisualizer } from "./components/RouteVisualizer";
import { QuantileRiskInspector } from "./components/QuantileRiskInspector";
import { AgentShowdown } from "./components/AgentShowdown";
import { TrainingDiagnostics } from "./components/TrainingDiagnostics";
import { Overview } from "./components/Overview";
import { fetchSimulation, fetchShowdown } from "./api";
import type { SimulationRequest, SimulationResponse, ShowdownResponse, ShowdownRequest } from "./types";
import { DEFAULT_INTERNATIONAL_ROUTE, type InternationalRoute } from "./config/internationalRoutes";

const defaultRequest: SimulationRequest = { urgency_tier: 1, shipment_weight: 15, deadline_remaining: 12, disruption_shock_prob: 0.35, seed: 42, agent_id: "qr_dqn_cvar", volume: 30, dest_inventory: 20 };

export function App() {
  const [activePage, setActivePage] = useState<AppPage>("overview");
  const [serverOnline, setServerOnline] = useState(true);
  const [simRequest, setSimRequest] = useState<SimulationRequest>(defaultRequest);
  const [internationalRoute, setInternationalRoute] = useState<InternationalRoute>(DEFAULT_INTERNATIONAL_ROUTE);
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playbackTimerRef = useRef<number | null>(null);
  const [showdownData, setShowdownData] = useState<ShowdownResponse | null>(null);
  const [showdownLoading, setShowdownLoading] = useState(false);
  const handleSimulate = async (customReq?: SimulationRequest) => { const req = customReq || simRequest; setSimLoading(true); setSimError(null); setIsPlaying(false); try { setSimulation(await fetchSimulation(req)); setCurrentStepIndex(0); setServerOnline(true); } catch (err: unknown) { setSimError(err instanceof Error ? err.message : "Failed to execute simulation"); setServerOnline(false); } finally { setSimLoading(false); } };
  const handleRunShowdown = async () => { setShowdownLoading(true); try { const { agent_id, c_air, c_ocean, c_land, ...showReq } = simRequest; void agent_id; void c_air; void c_ocean; void c_land; setShowdownData(await fetchShowdown(showReq as ShowdownRequest)); setServerOnline(true); } catch { setServerOnline(false); } finally { setShowdownLoading(false); } };
  const handleSelectAgentForSim = (agentId: string) => { const request = { ...simRequest, agent_id: agentId }; setSimRequest(request); setActivePage("simulator"); handleSimulate(request); };
  useEffect(() => { handleSimulate(); }, []);
  useEffect(() => { if (!isPlaying || !simulation) return; playbackTimerRef.current = window.setInterval(() => setCurrentStepIndex(prev => { if (prev < simulation.trajectory.length - 1) return prev + 1; setIsPlaying(false); return prev; }), 1800); return () => { if (playbackTimerRef.current) clearInterval(playbackTimerRef.current); }; }, [isPlaying, simulation]);
  const currentStep = simulation?.trajectory[currentStepIndex] || null;
  const titles: Record<AppPage,string> = { overview:"", simulator:"Live route simulator", showdown:"Multi-agent benchmark", quantiles:"Quantile & CVaR inspector", training:"Training diagnostics" };
  return <div className="research-shell min-h-screen"><Navbar activePage={activePage} setActivePage={setActivePage} serverOnline={serverOnline}/><main className="app-main">{activePage !== "overview" && <div className="page-intro"><div><p className="eyebrow">Global relief corridor simulation</p><h1>{titles[activePage]}</h1></div><span className="scenario-chip">Seed {simRequest.seed} · {simRequest.urgency_tier === 1 ? "Critical relief" : "Standard relief"}</span></div>}{simError && <div className="error-state"><AlertCircle size={18}/><span>{simError}</span><button onClick={() => handleSimulate()}><RefreshCw size={14}/> Retry</button></div>}{activePage === "overview" && <Overview onNavigate={setActivePage}/>} {activePage === "simulator" && <div className="simulator-layout"><aside className="control-drawer"><ControlSidebar request={simRequest} setRequest={setSimRequest} route={internationalRoute} setRoute={setInternationalRoute} onSimulate={() => handleSimulate()} loading={simLoading} trajectory={simulation?.trajectory || []} currentStepIndex={currentStepIndex} setCurrentStepIndex={setCurrentStepIndex} isPlaying={isPlaying} setIsPlaying={setIsPlaying}/></aside><section className="workspace"><Kpis simulation={simulation}/><RouteVisualizer simulation={simulation} currentStepIndex={currentStepIndex} route={internationalRoute}/><DecisionLog simulation={simulation}/></section></div>} {activePage === "showdown" && <AgentShowdown showdownData={showdownData} loading={showdownLoading} onRunShowdown={handleRunShowdown} onSelectAgentForSim={handleSelectAgentForSim}/>} {activePage === "quantiles" && <QuantileRiskInspector currentStep={currentStep} initialQuantiles={simulation?.trajectory[0]?.quantiles_analysis || null}/>} {activePage === "training" && <TrainingDiagnostics/>}</main></div>;
}
function Kpis({ simulation }: { simulation: SimulationResponse | null }) { const cards = [[CircleDollarSign,"Total freight cost",simulation ? `$${simulation.total_freight_cost.toLocaleString()}` : "—","Cost-optimized dispatch"],[CheckCircle2,"Constraint status",simulation?.status === "Delivered" ? "Feasible" : simulation ? "Violation" : "—",simulation?.status === "Delivered" ? "Within delivery SLA" : "Awaiting episode"],[PackageCheck,"Demand fulfilled",simulation ? (simulation.status === "Delivered" ? "100%" : "At risk") : "—","Destination service level"],[Clock3,"Episode decisions",simulation ? `${simulation.trajectory.length} steps` : "—",simulation ? `${simulation.total_days.toFixed(1)} transit days` : "Run a scenario"]] as const; return <div className="kpi-grid">{cards.map(([Icon,label,value,sub]) => <div className="kpi-card" key={label}><Icon size={17}/><span>{label}</span><strong>{value}</strong><small>{sub}</small></div>)}</div>; }
function DecisionLog({ simulation }: { simulation: SimulationResponse | null }) { if (!simulation) return <div className="decision-log skeleton-card"><BarChart3 size={18}/> Loading episode telemetry…</div>; return <details className="decision-log"><summary>Decision trace <span>{simulation.trajectory.length} dispatch decisions</span></summary><div className="table-wrap"><table><thead><tr><th>Step</th><th>Transition</th><th>Action</th><th>Mode</th><th>Cost</th><th>Deadline</th></tr></thead><tbody>{simulation.trajectory.map(s => <tr key={s.step_index}><td>{s.step_index + 1}</td><td>{s.location_start_name} → {s.location_end_name}</td><td>{s.action_name}</td><td>{s.transport_mode}</td><td>${s.freight_cost.toLocaleString()}</td><td>{s.deadline_remaining.toFixed(1)} d</td></tr>)}</tbody></table></div></details>; }
export default App;
