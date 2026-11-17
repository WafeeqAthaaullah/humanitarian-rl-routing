import { load, registerLoaders } from "@loaders.gl/core";
import { GLTFLoader as LoadersGLTFLoader } from "@loaders.gl/gltf";
import { useEffect, useState } from "react";
import { GLTFLoader as ThreeGLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
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
    Promise.all(Object.entries(MODEL_CONFIG).map(async ([key, config]) => {
      try {
        const [threeGltf, scenegraph] = await Promise.all([
          threeLoader.loadAsync(config.url),
          load(config.url, LoadersGLTFLoader),
        ]);
        const bounds = new THREE.Box3().setFromObject(threeGltf.scene);
        const size = bounds.getSize(new THREE.Vector3());
        const longestAxis = Math.max(size.x, size.y, size.z, 0.001);
        return [key as Model3dKey, { scenegraph, baseScale: 1 / longestAxis, loaded: true }] as const;
      } catch {
        return [key as Model3dKey, fallbackModel(key as Model3dKey)] as const;
      }
    })).then(entries => {
      if (active) setModels(Object.fromEntries(entries) as Record<Model3dKey, LoadedModel>);
    });
    return () => { active = false; };
  }, []);

  return models;
}
