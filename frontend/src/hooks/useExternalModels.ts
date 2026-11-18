import { load, registerLoaders } from "@loaders.gl/core";
import { GLTFLoader as LoadersGLTFLoader } from "@loaders.gl/gltf";
import { useEffect, useState } from "react";
import { GLTFLoader as ThreeGLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import * as THREE from "three";
import { MODEL_CONFIG, type Model3dKey } from "../config/models3d";

registerLoaders([LoadersGLTFLoader]);

export interface LoadedModel {
  scenegraph: unknown;
  baseScale: number;
  loaded: boolean;
}

function fallbackModel(key: Model3dKey): LoadedModel {
  return { scenegraph: MODEL_CONFIG[key].url, baseScale: 1, loaded: false };
}

/**
 * Normalize a cloned GLTF scene so that:
 *   - It is scaled uniformly so the longest axis = 1
 *   - Its horizontal center sits at (0, 0) and its bottom (min.y) rests at y = 0
 *   - An inner yaw/pitch/roll rotation offset is applied so the model faces forward
 *   - PBR materials preserve their original GLTF textures, colorSpaces, and vertex-color settings
 *
 * Returns the outer THREE.Group (position/tangent driven) wrapping the inner mesh group.
 */
function normalizeModel(scene: THREE.Group, rotationOffset: [number, number, number]): { group: THREE.Group; baseScale: number } {
  // --- 1. Measure raw bounding box to find longest axis ---
  const rawBox = new THREE.Box3().setFromObject(scene);
  const rawSize = rawBox.getSize(new THREE.Vector3());
  const longestAxis = Math.max(rawSize.x, rawSize.y, rawSize.z, 0.001);
  const baseScale = 1 / longestAxis;

  // Apply uniform scale
  scene.scale.multiplyScalar(baseScale);
  scene.updateMatrixWorld(true);

  // --- 2. Re-measure after scaling and center horizontally, ground vertically ---
  const scaledBox = new THREE.Box3().setFromObject(scene);
  const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
  const scaledMin = scaledBox.min;

  // Shift so horizontal center is (0, 0) and bottom rests at y = 0
  scene.position.set(-scaledCenter.x, -scaledMin.y, -scaledCenter.z);
  scene.updateMatrixWorld(true);

  // --- 3. Apply per-model rotation offset to the inner mesh ---
  const innerGroup = new THREE.Group();
  innerGroup.name = "model-inner-rotation";
  // Rotation offset is [pitch, yaw, roll] in degrees; apply as Euler YXZ
  const [pitchDeg, yawDeg, rollDeg] = rotationOffset;
  innerGroup.rotation.set(
    THREE.MathUtils.degToRad(pitchDeg),
    THREE.MathUtils.degToRad(yawDeg),
    THREE.MathUtils.degToRad(rollDeg),
    "YXZ"
  );
  innerGroup.add(scene);

  // --- 4. Outer group for route position/tangent orientation ---
  const outerGroup = new THREE.Group();
  outerGroup.name = "model-outer-transform";
  outerGroup.add(innerGroup);

  return { group: outerGroup, baseScale };
}

/**
 * Walk every material in the scene and ensure PBR textures render correctly:
 *   - Preserve SRGBColorSpace on color/emissive maps
 *   - Disable vertexColors if the geometry has no color attribute
 *   - Do NOT override/replace original GLTF materials
 */
function fixMaterials(scene: THREE.Object3D): void {
  scene.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    const geom = child.geometry as THREE.BufferGeometry | undefined;
    const hasVertexColors = !!geom?.getAttribute("color");

    for (const mat of materials) {
      if (!(mat instanceof THREE.Material)) continue;

      // Disable vertexColors if geometry has no color attribute
      if ("vertexColors" in mat && (mat as THREE.MeshStandardMaterial).vertexColors && !hasVertexColors) {
        (mat as THREE.MeshStandardMaterial).vertexColors = false;
        mat.needsUpdate = true;
      }

      // Ensure color maps use SRGB
      if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshPhysicalMaterial) {
        if (mat.map) mat.map.colorSpace = THREE.SRGBColorSpace;
        if (mat.emissiveMap) mat.emissiveMap.colorSpace = THREE.SRGBColorSpace;
        // Don't touch metalness/roughness maps — they should stay Linear
      }
    }
  });
}

/** Shared environment map for PBR models — created once per renderer */
let _envTexture: THREE.Texture | null = null;
function getOrCreateEnvironmentMap(renderer: THREE.WebGLRenderer): THREE.Texture {
  if (_envTexture) return _envTexture;
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  pmremGenerator.compileEquirectangularShader();
  const envScene = new RoomEnvironment();
  _envTexture = pmremGenerator.fromScene(envScene, 0.04).texture;
  pmremGenerator.dispose();
  return _envTexture;
}

/**
 * Pre-bake an environment map into every material in the scene so metallic PBR surfaces
 * render their true colors and reflections (instead of appearing black/grey without lighting).
 *
 * Uses a small offscreen renderer to generate the PMREM texture.
 */
function bakeEnvironmentMap(scene: THREE.Object3D): void {
  // Create a small offscreen renderer just for PMREM generation
  const offscreenRenderer = new THREE.WebGLRenderer({ antialias: false });
  offscreenRenderer.setSize(1, 1); // minimal — we only need the PMREM pipeline
  offscreenRenderer.outputColorSpace = THREE.SRGBColorSpace;
  offscreenRenderer.toneMapping = THREE.ACESFilmicToneMapping;

  const envMap = getOrCreateEnvironmentMap(offscreenRenderer);

  scene.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const mat of materials) {
      if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshPhysicalMaterial) {
        mat.envMap = envMap;
        mat.envMapIntensity = 1.0;
        mat.needsUpdate = true;
      }
    }
  });

  offscreenRenderer.dispose();
}

export function useExternalModels() {
  const [models, setModels] = useState<Record<Model3dKey, LoadedModel>>(() => ({
    cargoShip: fallbackModel("cargoShip"),
    cargoPlane: fallbackModel("cargoPlane"),
    cargoTruck: fallbackModel("cargoTruck"),
    isoContainer: fallbackModel("isoContainer"),
  }));

  useEffect(() => {
    let active = true;
    const threeLoader = new ThreeGLTFLoader();

    Promise.all(
      Object.entries(MODEL_CONFIG).map(async ([key, config]) => {
        try {
          const [threeGltf, scenegraph] = await Promise.all([
            threeLoader.loadAsync(config.url),
            load(config.url, LoadersGLTFLoader),
          ]);

          // --- Fix materials BEFORE measuring (preserves original GLTF data) ---
          fixMaterials(threeGltf.scene);

          // --- Bake environment map so metallic PBR surfaces show true colors ---
          bakeEnvironmentMap(threeGltf.scene);

          // --- Normalize geometry: center, ground, apply rotation offset ---
          const { baseScale } = normalizeModel(
            threeGltf.scene,
            config.rotationOffset
          );

          return [
            key as Model3dKey,
            { scenegraph, baseScale, loaded: true },
          ] as const;
        } catch (err) {
          console.warn(`[useExternalModels] Failed to load ${key}:`, err);
          return [key as Model3dKey, fallbackModel(key as Model3dKey)] as const;
        }
      })
    ).then((entries) => {
      if (active)
        setModels(
          Object.fromEntries(entries) as Record<Model3dKey, LoadedModel>
        );
    });

    return () => {
      active = false;
    };
  }, []);

  return models;
}
