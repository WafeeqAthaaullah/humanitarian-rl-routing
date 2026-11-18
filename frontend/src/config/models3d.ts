export type Model3dKey = "cargoShip" | "cargoPlane" | "cargoTruck" | "isoContainer";

export interface Model3dConfig {
  url: string;
  yawOffsetDeg: number;
  basePitchDeg: number;
  scaleMultiplier: number;
  altitudeOffsetM: number;
  sizeScale: number;
  sizeMinPixels: number;
  sizeMaxPixels: number;
  /**
   * Per-model rotation offset (in degrees) applied to the INNER mesh
   * so the model faces forward (+Y in deck.gl) along its travel vector.
   * [pitch, yaw, roll] – most models only need a yaw correction.
   */
  rotationOffset: [number, number, number];
}

export const MODEL_CONFIG: Record<Model3dKey, Model3dConfig> = {
  cargoShip: {
    url: "/models/Container Ship.glb",
    yawOffsetDeg: 0,
    basePitchDeg: 90,
    scaleMultiplier: 1,
    altitudeOffsetM: 0,
    sizeScale: 32000,
    sizeMinPixels: 34,
    sizeMaxPixels: 68,
    // Ship models typically face +X or -X in their local space; rotate 90° so bow points forward
    rotationOffset: [0, -90, 0],
  },
  cargoPlane: {
    url: "/models/Airplane.glb",
    yawOffsetDeg: 0,
    basePitchDeg: 90,
    scaleMultiplier: 1,
    altitudeOffsetM: 120,
    sizeScale: 28000,
    sizeMinPixels: 32,
    sizeMaxPixels: 64,
    // Airplane nose typically points +Z or +X; rotate so nose faces forward
    rotationOffset: [0, 0, 0],
  },
  cargoTruck: {
    url: "/models/Cargo Truck.glb",
    yawOffsetDeg: 0,
    basePitchDeg: 90,
    scaleMultiplier: 1,
    altitudeOffsetM: 0,
    sizeScale: 22000,
    sizeMinPixels: 28,
    sizeMaxPixels: 56,
    // Truck cab typically faces +Z; rotate so it faces forward
    rotationOffset: [0, 180, 0],
  },
  isoContainer: {
    url: "/models/Shipping Container.glb",
    yawOffsetDeg: 0,
    basePitchDeg: 90,
    scaleMultiplier: 1,
    altitudeOffsetM: 0,
    sizeScale: 16000,
    sizeMinPixels: 18,
    sizeMaxPixels: 36,
    rotationOffset: [0, 0, 0],
  },
};
