import { useEffect, useMemo, useState } from "react";
import Map, { NavigationControl, useControl } from "@vis.gl/react-maplibre";
import { MapboxOverlay } from "@deck.gl/mapbox";
import type { Layer } from "@deck.gl/core";
import { ArcLayer, PathLayer } from "@deck.gl/layers";
import { ScenegraphLayer } from "@deck.gl/mesh-layers";
import "maplibre-gl/dist/maplibre-gl.css";
import { AlertTriangle, Box, Map as MapIcon, MapPinned, Radio, RotateCcw } from "lucide-react";
import type { SimulationResponse, TrajectoryStep } from "../types";
import { MODEL_CONFIG, type Model3dKey } from "../config/models3d";
import { useExternalModels } from "../hooks/useExternalModels";
import type { InternationalRoute, InternationalTransportMode } from "../config/internationalRoutes";

interface RouteVisualizerProps {
  simulation: SimulationResponse | null;
  currentStepIndex: number;
  route: InternationalRoute;
}

type Position = [number, number, number];
type GeoPoint = [number, number];
type VehicleDatum = { position: Position; orientation: [number, number, number] };

const HORMUZ: GeoPoint = [56.45, 26.55];
const RAS_AL_HADD: GeoPoint = [59.85, 22.5];
const SALALAH_OFFSHORE: GeoPoint = [54.5, 16.5];
const GULF_OF_ADEN: GeoPoint = [46.5, 12.6];
const BAB_EL_MANDEB: GeoPoint = [43.35, 12.55];
const CAPE_GUARDAFUI: GeoPoint = [51.4, 11.8];
const EAST_ARABIAN_SEA: GeoPoint = [63.5, 22.5];
const BAY_OF_BENGAL: GeoPoint = [86, 15];

function DeckOverlay({ layers }: { layers: Layer[] }) {
  const overlay = useControl(() => new MapboxOverlay({ interleaved: false, layers }));
  useEffect(() => overlay.setProps({ layers }), [layers, overlay]);
  return null;
}

function lineForMode(route: InternationalRoute, mode: Exclude<InternationalTransportMode, "multimodal">, from: GeoPoint, to: GeoPoint): GeoPoint[] {
  if (mode !== "ocean") return [from, to];
  const isGlobalOriginLeg = from[0] > 50 && from[1] > 15;
  if (!isGlobalOriginLeg) {
    if (route.checkpoint.id === "mombasa" || route.destination.id === "dadaab") return [from, CAPE_GUARDAFUI, to];
    if (route.checkpoint.id === "karachi" || route.destination.id === "dhaka" || route.destination.id === "kathmandu") return [from, BAY_OF_BENGAL, to];
    return [from, BAB_EL_MANDEB, to];
  }
  if (route.checkpoint.id === "mombasa" || route.destination.id === "dadaab") {
    return [from, HORMUZ, RAS_AL_HADD, SALALAH_OFFSHORE, CAPE_GUARDAFUI, to];
  }
  if (route.checkpoint.id === "karachi" || route.destination.id === "dhaka" || route.destination.id === "kathmandu") {
    return [from, HORMUZ, RAS_AL_HADD, EAST_ARABIAN_SEA, BAY_OF_BENGAL, to];
  }
  return [from, HORMUZ, RAS_AL_HADD, SALALAH_OFFSHORE, GULF_OF_ADEN, BAB_EL_MANDEB, to];
}

function interpolatePath(path: GeoPoint[], progress: number, altitude: number): Position {
  const safeProgress = Math.max(0, Math.min(1, progress));
  const segment = Math.min(path.length - 2, Math.floor(safeProgress * (path.length - 1)));
  const localProgress = safeProgress * (path.length - 1) - segment;
  const from = path[segment];
  const to = path[segment + 1];
  return [from[0] + (to[0] - from[0]) * localProgress, from[1] + (to[1] - from[1]) * localProgress, altitude];
}

function bearingFor(path: GeoPoint[], progress: number): number {
  const point = interpolatePath(path, progress, 0);
  const next = interpolatePath(path, Math.min(1, progress + 0.02), 0);
  return Math.atan2(next[0] - point[0], next[1] - point[1]) * 180 / Math.PI;
}

function modeForStep(step: TrajectoryStep | undefined, route: InternationalRoute): Exclude<InternationalTransportMode, "multimodal"> {
  if (route.mode !== "multimodal") return route.mode;
  const backendMode = step?.transport_mode.toLowerCase() || "";
  if (backendMode.includes("air")) return "air";
  if (backendMode.includes("ocean") || backendMode.includes("sea")) return "ocean";
  return step ? "overland" : route.leg1Mode;
}

function fitCamera(route: InternationalRoute, view: "3d" | "2d") {
  const points = [route.origin.coordinates, route.checkpoint.coordinates, route.destination.coordinates];
  const minLon = Math.min(...points.map(point => point[0]));
  const maxLon = Math.max(...points.map(point => point[0]));
  const minLat = Math.min(...points.map(point => point[1]));
  const maxLat = Math.max(...points.map(point => point[1]));
  const longitude = (minLon + maxLon) / 2;
  const latitude = (minLat + maxLat) / 2;
  const span = Math.max(maxLon - minLon, maxLat - minLat, 8);
  const zoom = Math.max(3.2, Math.min(7.5, Math.log2(360 / span) - 0.35));
  return { longitude, latitude, zoom, pitch: view === "3d" ? 45 : 0, bearing: view === "3d" ? -10 : 0, padding: { top: 90, bottom: 90, left: 90, right: 90 } };
}

export function RouteVisualizer({ simulation, currentStepIndex, route }: RouteVisualizerProps) {
  const models = useExternalModels();
  const [view, setView] = useState<"3d" | "2d">("3d");
  const [preset, setPreset] = useState("Corridor Overview");
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [calibration, setCalibration] = useState<Record<Model3dKey, { pitch: number; yaw: number; scale: number }>>({
    cargoShip: { pitch: 0, yaw: 0, scale: 1 },
    cargoPlane: { pitch: 0, yaw: 0, scale: 1 },
    cargoTruck: { pitch: 0, yaw: 0, scale: 1 },
    isoContainer: { pitch: 0, yaw: 0, scale: 1 },
  });
  const [progress, setProgress] = useState(0);
  const step = simulation?.trajectory[currentStepIndex];
  const activeMode = modeForStep(step, route);
  const leg1Path = lineForMode(route, route.mode === "multimodal" ? route.leg1Mode : route.mode, route.origin.coordinates, route.checkpoint.coordinates);
  const leg2Path = lineForMode(route, route.mode === "multimodal" ? route.leg2Mode : route.mode, route.checkpoint.coordinates, route.destination.coordinates);

  useEffect(() => {
    const target = simulation?.trajectory.length ? currentStepIndex / Math.max(1, simulation.trajectory.length - 1) : 0;
    const start = progress;
    const started = performance.now();
    let frame = 0;
    const animate = (now: number) => {
      const eased = Math.min(1, (now - started) / 550);
      setProgress(start + (target - start) * (eased * (2 - eased)));
      if (eased < 1) frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [currentStepIndex, simulation?.trajectory.length]);

  const layers = useMemo(() => {
    const activePath = activeMode === "ocean" ? (progress < 0.5 ? leg1Path : leg2Path) : activeMode === "air" ? [route.origin.coordinates, route.checkpoint.coordinates, route.destination.coordinates] : [route.origin.coordinates, route.checkpoint.coordinates, route.destination.coordinates];
    const activeLocalProgress = progress < 0.5 ? progress * 2 : (progress - 0.5) * 2;
    const activePosition = interpolatePath(activePath, activeLocalProgress, activeMode === "air" ? 120000 : 120);
    const activeBearing = bearingFor(activePath, activeLocalProgress);
    const layerForVehicle = (key: Model3dKey, data: VehicleDatum[]) => new ScenegraphLayer({
      id: `${key}-international`,
      data,
      scenegraph: models[key].scenegraph,
      getPosition: datum => datum.position,
      getOrientation: datum => [MODEL_CONFIG[key].basePitchDeg + datum.orientation[0] + calibration[key].pitch, -datum.orientation[1] + MODEL_CONFIG[key].yawOffsetDeg + calibration[key].yaw, datum.orientation[2]],
      getScale: [models[key].baseScale * MODEL_CONFIG[key].scaleMultiplier * calibration[key].scale, models[key].baseScale * MODEL_CONFIG[key].scaleMultiplier * calibration[key].scale, models[key].baseScale * MODEL_CONFIG[key].scaleMultiplier * calibration[key].scale],
      sizeScale: MODEL_CONFIG[key].sizeScale,
      sizeMinPixels: MODEL_CONFIG[key].sizeMinPixels,
      sizeMaxPixels: MODEL_CONFIG[key].sizeMaxPixels,
      getColor: [255, 255, 255],
      pickable: true,
    });
    const containerData = [route.origin, route.checkpoint, route.destination].flatMap((waypoint, hubIndex) => Array.from({ length: 4 }, (_, stack) => ({
      position: [waypoint.coordinates[0], waypoint.coordinates[1], stack * 120] as Position,
      color: hubIndex === 2 && step?.tier1_violation ? [220, 38, 38] : hubIndex === 1 ? [217, 119, 6] : [13, 148, 136],
    })));
    const oceanPath = route.mode === "ocean" || route.mode === "multimodal" && (route.leg1Mode === "ocean" || route.leg2Mode === "ocean") ? [...(route.leg1Mode === "ocean" || route.mode === "ocean" ? leg1Path : []), ...(route.leg2Mode === "ocean" || route.mode === "ocean" ? leg2Path : [])] : [];
    const roadPath = route.mode === "overland" || route.mode === "multimodal" && (route.leg1Mode === "overland" || route.leg2Mode === "overland") ? [route.origin.coordinates, route.checkpoint.coordinates, route.destination.coordinates] : [];
    const airSegments = route.mode === "air" ? [[route.origin.coordinates, route.checkpoint.coordinates], [route.checkpoint.coordinates, route.destination.coordinates]] : route.mode === "multimodal" ? [[route.origin.coordinates, route.checkpoint.coordinates], [route.checkpoint.coordinates, route.destination.coordinates]].filter((_, index) => [route.leg1Mode, route.leg2Mode][index] === "air") : [];
    return [
      new PathLayer({ id: "international-sea-lanes", data: oceanPath.length ? [{ path: oceanPath.map(point => [point[0], point[1], 0] as Position) }] : [], getPath: datum => datum.path, getColor: [2, 132, 199], getWidth: 9, widthMinPixels: 3, pickable: true }),
      new PathLayer({ id: "international-road-corridors", data: roadPath.length ? [{ path: roadPath.map(point => [point[0], point[1], 100] as Position) }] : [], getPath: datum => datum.path, getColor: [217, 119, 6], getWidth: 8, widthMinPixels: 3, pickable: true }),
      new ArcLayer({ id: "international-air-corridors", data: airSegments.map(([from, to]) => ({ source: [...from, 0] as Position, target: [...to, 120000] as Position })), getSourcePosition: datum => datum.source, getTargetPosition: datum => datum.target, getSourceColor: [79, 70, 229], getTargetColor: [124, 58, 237], getWidth: 7, getHeight: 0.65, pickable: true }),
      new ScenegraphLayer({ id: "international-container-yards", data: containerData, scenegraph: models.isoContainer.scenegraph, getPosition: datum => datum.position, getColor: datum => new Uint8Array(datum.color), getOrientation: () => [MODEL_CONFIG.isoContainer.basePitchDeg + calibration.isoContainer.pitch, calibration.isoContainer.yaw, 0], getScale: [models.isoContainer.baseScale * calibration.isoContainer.scale, models.isoContainer.baseScale * calibration.isoContainer.scale, models.isoContainer.baseScale * calibration.isoContainer.scale], sizeScale: MODEL_CONFIG.isoContainer.sizeScale, sizeMinPixels: MODEL_CONFIG.isoContainer.sizeMinPixels, sizeMaxPixels: MODEL_CONFIG.isoContainer.sizeMaxPixels, pickable: true }),
      layerForVehicle("cargoShip", [{ position: [...route.origin.coordinates, 400] as Position, orientation: [0, bearingFor(leg1Path, 0), 0] }, { position: [...route.checkpoint.coordinates, 400] as Position, orientation: [0, bearingFor(leg2Path, 0), 0] }, ...(activeMode === "ocean" ? [{ position: activePosition, orientation: [0, activeBearing, 0] as [number, number, number] }] : [])]),
      layerForVehicle("cargoPlane", [{ position: [...route.origin.coordinates, 120000] as Position, orientation: [8, 45, 0] }, { position: [...route.checkpoint.coordinates, 120000] as Position, orientation: [8, 45, 0] }, ...(activeMode === "air" ? [{ position: activePosition, orientation: [8, activeBearing, 0] as [number, number, number] }] : [])]),
      layerForVehicle("cargoTruck", [{ position: [...route.checkpoint.coordinates, 300] as Position, orientation: [0, 90, 0] }, { position: [...route.destination.coordinates, 300] as Position, orientation: [0, 90, 0] }, ...(activeMode === "overland" ? [{ position: activePosition, orientation: [0, activeBearing, 0] as [number, number, number] }] : [])]),
    ] as Layer[];
  }, [activeMode, calibration, leg1Path, leg2Path, models, progress, route, step]);

  const camera = preset === "Dubai Origin Hub" ? { ...fitCamera(route, view), longitude: route.origin.coordinates[0], latitude: route.origin.coordinates[1], zoom: 8 } : preset === "Regional Checkpoint" ? { ...fitCamera(route, view), longitude: route.checkpoint.coordinates[0], latitude: route.checkpoint.coordinates[1], zoom: 8 } : preset === "Crisis Destination" ? { ...fitCamera(route, view), longitude: route.destination.coordinates[0], latitude: route.destination.coordinates[1], zoom: 8 } : fitCamera(route, view);
  const adjustCalibration = (key: Model3dKey, field: "pitch" | "yaw" | "scale", value: number) => {
    setCalibration(previous => ({ ...previous, [key]: { ...previous[key], [field]: field === "scale" ? Math.max(0.5, Math.min(2, value)) : value } }));
  };

  return <section className="map-card">
    <header className="map-toolbar"><div><p className="eyebrow">Global relief corridor · live digital twin</p><h2><Radio size={16} /> International 3D Geospatial Twin</h2></div><div className="map-actions"><label className="preset-select"><MapPinned size={14} /><select value={preset} onChange={event => setPreset(event.target.value)}>{["Corridor Overview", "Dubai Origin Hub", "Regional Checkpoint", "Crisis Destination"].map(item => <option key={item}>{item}</option>)}</select></label><div className="segmented"><button className={view === "3d" ? "active" : ""} onClick={() => setView("3d")}><Box size={13} /> 3D Twin</button><button className={view === "2d" ? "active" : ""} onClick={() => setView("2d")}><MapIcon size={13} /> 2D Graph</button></div><button className="icon-button" title="Reset view" onClick={() => setPreset("Corridor Overview")}><RotateCcw size={14} /></button><button className="calibration-button" onClick={() => setCalibrationOpen(!calibrationOpen)}>3D calibration</button></div>{calibrationOpen && <div className="calibration-popover"><strong>Model calibration</strong>{(["cargoShip", "cargoPlane", "cargoTruck", "isoContainer"] as const).map(key => <div key={key}><span>{key.replace("cargo", "")}</span><button onClick={() => adjustCalibration(key, "pitch", 0)}>Pitch 0°</button><button onClick={() => adjustCalibration(key, "pitch", 90)}>Pitch 90°</button><button onClick={() => adjustCalibration(key, "pitch", -90)}>Pitch -90°</button><button onClick={() => adjustCalibration(key, "yaw", -90)}>Yaw -90°</button><button onClick={() => adjustCalibration(key, "yaw", 90)}>Yaw +90°</button><button onClick={() => adjustCalibration(key, "yaw", 180)}>Yaw 180°</button><button onClick={() => adjustCalibration(key, "scale", calibration[key].scale - 0.1)}>Scale −</button><button onClick={() => adjustCalibration(key, "scale", calibration[key].scale + 0.1)}>Scale +</button><small>{calibration[key].scale.toFixed(1)}x</small></div>)}</div>}</header>
    <div className="map-viewport"><Map key={`${route.origin.id}-${route.checkpoint.id}-${route.destination.id}-${preset}-${view}`} initialViewState={camera} mapStyle="https://tiles.openfreemap.org/styles/liberty" attributionControl={false}><NavigationControl position="bottom-right" showCompass /><DeckOverlay layers={layers} /></Map><div className="international-map-label"><strong>{route.origin.label.split(",")[0]}</strong><span>→ {route.checkpoint.label.split(",")[0]} → {route.destination.label.split(",")[0]}</span><small>{route.mode === "multimodal" ? `${route.leg1Mode} / ${route.leg2Mode}` : route.mode} · progress {(progress * 100).toFixed(0)}%</small></div>{step?.disruption_triggered && <div className="map-alert"><AlertTriangle size={14} /> Stochastic disruption active on current leg</div>}</div>
    <footer className="map-legend"><span><i className="legend-line air" /> Strategic airlift</span><span><i className="legend-line road" /> Cross-border convoy</span><span><i className="legend-line sea" /> Maritime sea lane</span><span><i className="legend-square healthy" /> Global staging stock</span><span className="map-source">OpenFreeMap Liberty · International corridor projection</span></footer>
  </section>;
}
