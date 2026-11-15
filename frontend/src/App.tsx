import { useState, useEffect, useRef } from "react";
import { Navbar } from "./components/Navbar";
import { ControlSidebar } from "./components/ControlSidebar";
import { RouteVisualizer } from "./components/RouteVisualizer";
import { QuantileRiskInspector } from "./components/QuantileRiskInspector";
import { AgentShowdown } from "./components/AgentShowdown";
import { TrainingDiagnostics } from "./components/TrainingDiagnostics";
import { fetchSimulation, fetchShowdown } from "./api";
import type {
  SimulationRequest,
  SimulationResponse,
  ShowdownResponse,
  ShowdownRequest,
} from "./types";
import { AlertCircle } from "lucide-react";

export function App() {
  const [activeTab, setActiveTab] = useState<
    "simulator" | "showdown" | "quantiles" | "training"
  >("simulator");
  const [serverOnline, setServerOnline] = useState<boolean>(true);

  // Simulation State
  const [simRequest, setSimRequest] = useState<SimulationRequest>({
    urgency_tier: 1.0, // Critical
    shipment_weight: 15.0, // 15 tons
    deadline_remaining: 12.0, // 12 days
    disruption_shock_prob: 0.35, // 35% maritime bottleneck probability
    seed: 42,
    agent_id: "qr_dqn_cvar",
    volume: 30.0,
    dest_inventory: 20.0,
  });

  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);
  const [simLoading, setSimLoading] = useState<boolean>(false);
  const [simError, setSimError] = useState<string | null>(null);

  // Playback state
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playbackTimerRef = useRef<number | null>(null);

  // Showdown State
  const [showdownData, setShowdownData] = useState<ShowdownResponse | null>(
    null
  );
  const [showdownLoading, setShowdownLoading] = useState<boolean>(false);

  // Run simulation handler
  const handleSimulate = async (customReq?: SimulationRequest) => {
    const req = customReq || simRequest;
    setSimLoading(true);
    setSimError(null);
    setIsPlaying(false);
    try {
      const data = await fetchSimulation(req);
      setSimulation(data);
      setCurrentStepIndex(0);
      setServerOnline(true);
    } catch (err: any) {
      setSimError(err.message || "Failed to execute simulation");
      setServerOnline(false);
    } finally {
      setSimLoading(false);
    }
  };

  // Run showdown handler
  const handleRunShowdown = async () => {
    setShowdownLoading(true);
    try {
      const showReq: ShowdownRequest = {
        urgency_tier: simRequest.urgency_tier,
        shipment_weight: simRequest.shipment_weight,
        deadline_remaining: simRequest.deadline_remaining,
        disruption_shock_prob: simRequest.disruption_shock_prob,
        seed: simRequest.seed,
        volume: simRequest.volume,
        dest_inventory: simRequest.dest_inventory,
      };
      const data = await fetchShowdown(showReq);
      setShowdownData(data);
      setServerOnline(true);
    } catch (err: any) {
      console.error("Showdown failed:", err);
    } finally {
      setShowdownLoading(false);
    }
  };

  // Inspect specific agent in simulator from Showdown
  const handleSelectAgentForSim = (agentId: string) => {
    const updated = { ...simRequest, agent_id: agentId };
    setSimRequest(updated);
    setActiveTab("simulator");
    handleSimulate(updated);
  };

  // Auto-run initial simulation on mount
  useEffect(() => {
    handleSimulate();
  }, []);

  // Timeline playback timer
  useEffect(() => {
    if (!isPlaying || !simulation) {
      if (playbackTimerRef.current) {
        clearInterval(playbackTimerRef.current);
        playbackTimerRef.current = null;
      }
      return;
    }

    playbackTimerRef.current = window.setInterval(() => {
      setCurrentStepIndex((prev) => {
        if (prev < simulation.trajectory.length - 1) {
          return prev + 1;
        } else {
          setIsPlaying(false);
          return prev;
        }
      });
    }, 1800);

    return () => {
      if (playbackTimerRef.current) {
        clearInterval(playbackTimerRef.current);
      }
    };
  }, [isPlaying, simulation]);

  const currentStep = simulation?.trajectory[currentStepIndex] || null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        serverOnline={serverOnline}
      />

      {/* Main View Area */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 sm:p-6 flex flex-col lg:flex-row gap-6">
        {/* Left Persistent Control Sidebar */}
        <ControlSidebar
          request={simRequest}
          setRequest={setSimRequest}
          onSimulate={() => handleSimulate()}
          loading={simLoading}
          trajectory={simulation?.trajectory || []}
          currentStepIndex={currentStepIndex}
          setCurrentStepIndex={setCurrentStepIndex}
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
        />

        {/* Right Dynamic Content Pane */}
        <div className="flex-1 flex flex-col space-y-6 overflow-hidden">
          {simError && (
            <div className="p-4 rounded-xl bg-red-950/50 border border-red-500/50 text-red-200 text-xs flex items-center space-x-2">
              <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
              <span>{simError}</span>
            </div>
          )}

          {activeTab === "simulator" && (
            <div className="space-y-6">
              {/* Route & Pivot Visualizer */}
              <RouteVisualizer
                simulation={simulation}
                currentStepIndex={currentStepIndex}
              />

              {/* Live Quantile & CVaR Inspector below Route Visualizer */}
              <QuantileRiskInspector
                currentStep={currentStep}
                initialQuantiles={
                  simulation?.trajectory[0]?.quantiles_analysis || null
                }
              />
            </div>
          )}

          {activeTab === "showdown" && (
            <AgentShowdown
              showdownData={showdownData}
              loading={showdownLoading}
              onRunShowdown={handleRunShowdown}
              onSelectAgentForSim={handleSelectAgentForSim}
            />
          )}

          {activeTab === "quantiles" && (
            <div className="space-y-6">
              <QuantileRiskInspector
                currentStep={currentStep}
                initialQuantiles={
                  simulation?.trajectory[0]?.quantiles_analysis || null
                }
              />
            </div>
          )}

          {activeTab === "training" && <TrainingDiagnostics />}
        </div>
      </main>
    </div>
  );
}

export default App;
