import * as THREE from "three";

import { hidePlanet, planetEaseDwell, updatePlanetApproach, type PlanetApproachConfig } from "./planetApproach";

export { EARTH_CYCLE_SEC } from "./solarJourney";

/** Negative X = left side of screen, positive = right. */
export const EARTH_SIDE = -1;

const EARTH_APPROACH: PlanetApproachConfig = {
  zFar: -68,
  zNear: -18,
  xFar: 13,
  xNear: 6,
  yFar: 1.1,
  yNear: 0.04,
  scaleFar: 3.6,
  scaleNear: 8.4,
};

const TEXTURES = {
  map: "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
  bump: "https://unpkg.com/three-globe/example/img/earth-topology.png",
  specular: "https://unpkg.com/three-globe/example/img/earth-water.png",
  clouds: "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_clouds_1024.png",
  moon: "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg",
} as const;

const ATMOSPHERE_VERTEX = `
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const ATMOSPHERE_FRAGMENT = `
  varying vec3 vNormal;
  void main() {
    float intensity = pow(0.62 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 4.0);
    gl_FragColor = vec4(0.28, 0.55, 1.0, 1.0) * intensity;
  }
`;

export type EarthSystem = {
  root: THREE.Group;
  earthMesh: THREE.Mesh;
  cloudsMesh: THREE.Mesh;
  atmosphereMesh: THREE.Mesh;
  moonPivot: THREE.Group;
  moonMesh: THREE.Mesh;
  dispose: () => void;
};

export function createEarthSystem(loader: THREE.TextureLoader): Promise<EarthSystem> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = window.setTimeout(() => {
      if (!settled) reject(new Error("Earth textures timed out"));
    }, 45000);

    const onReady = (
      earthMap: THREE.Texture,
      earthBump: THREE.Texture,
      earthSpecular: THREE.Texture,
      cloudMap: THREE.Texture,
      moonMap: THREE.Texture,
    ) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);

      for (const tex of [earthMap, earthBump, earthSpecular, cloudMap, moonMap]) {
        tex.colorSpace = THREE.SRGBColorSpace;
      }

      const root = new THREE.Group();
      root.rotation.z = (23.5 * Math.PI) / 180;

      const earthGeo = new THREE.SphereGeometry(1, 48, 48);
      const earthMat = new THREE.MeshPhongMaterial({
        map: earthMap,
        bumpMap: earthBump,
        bumpScale: 0.05,
        specularMap: earthSpecular,
        specular: new THREE.Color("grey"),
        shininess: 10,
        fog: false,
        transparent: true,
        opacity: 1,
      });
      const earthMesh = new THREE.Mesh(earthGeo, earthMat);
      root.add(earthMesh);

      const cloudGeo = new THREE.SphereGeometry(1.02, 48, 48);
      const cloudMat = new THREE.MeshLambertMaterial({
        map: cloudMap,
        transparent: true,
        opacity: 0.88,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        fog: false,
      });
      const cloudsMesh = new THREE.Mesh(cloudGeo, cloudMat);
      root.add(cloudsMesh);

      const atmosphereGeo = new THREE.SphereGeometry(1.22, 48, 48);
      const atmosphereMat = new THREE.ShaderMaterial({
        vertexShader: ATMOSPHERE_VERTEX,
        fragmentShader: ATMOSPHERE_FRAGMENT,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        transparent: true,
        opacity: 1,
        depthWrite: false,
        fog: false,
      });
      const atmosphereMesh = new THREE.Mesh(atmosphereGeo, atmosphereMat);

      const moonPivot = new THREE.Group();
      root.add(moonPivot);
      const moonGeo = new THREE.SphereGeometry(0.27, 32, 32);
      const moonMat = new THREE.MeshPhongMaterial({ map: moonMap, fog: false, transparent: true, opacity: 1 });
      const moonMesh = new THREE.Mesh(moonGeo, moonMat);
      moonMesh.position.set(2.75, 0.35, 0.15);
      moonPivot.add(moonMesh);

      root.visible = false;
      atmosphereMesh.visible = false;

      const dispose = () => {
        earthGeo.dispose();
        cloudGeo.dispose();
        atmosphereGeo.dispose();
        moonGeo.dispose();
        earthMat.dispose();
        cloudMat.dispose();
        atmosphereMat.dispose();
        moonMat.dispose();
        earthMap.dispose();
        earthBump.dispose();
        earthSpecular.dispose();
        cloudMap.dispose();
        moonMap.dispose();
      };

      resolve({ root, earthMesh, cloudsMesh, atmosphereMesh, moonPivot, moonMesh, dispose });
    };

    const textures: THREE.Texture[] = [];
    let loaded = 0;
    const keys = ["map", "bump", "specular", "clouds", "moon"] as const;
    const onOneLoad = () => {
      loaded += 1;
      if (loaded === keys.length) {
        onReady(textures[0], textures[1], textures[2], textures[3], textures[4]);
      }
    };
    const onOneError = () => {
      if (!settled) {
        settled = true;
        window.clearTimeout(timeout);
        reject(new Error("Earth texture load failed"));
      }
    };

    loader.load(TEXTURES.map, (t) => { textures[0] = t; onOneLoad(); }, undefined, onOneError);
    loader.load(TEXTURES.bump, (t) => { textures[1] = t; onOneLoad(); }, undefined, onOneError);
    loader.load(TEXTURES.specular, (t) => { textures[2] = t; onOneLoad(); }, undefined, onOneError);
    loader.load(TEXTURES.clouds, (t) => { textures[3] = t; onOneLoad(); }, undefined, onOneError);
    loader.load(TEXTURES.moon, (t) => { textures[4] = t; onOneLoad(); }, undefined, onOneError);
  });
}

export function updateEarthApproach(
  system: EarthSystem,
  cycleLinear: number,
  dt: number,
  side: number = EARTH_SIDE,
): number {
  return updatePlanetApproach(
    system.root,
    system.atmosphereMesh,
    cycleLinear,
    dt,
    side,
    [
      { mesh: system.earthMesh, speed: 0.09 },
      { mesh: system.cloudsMesh, speed: 0.115 },
    ],
    [{ pivot: system.moonPivot, speed: 0.35 }],
    [
      { material: system.earthMesh.material as THREE.Material, baseOpacity: 1 },
      { material: system.cloudsMesh.material as THREE.Material, baseOpacity: 0.88 },
      { material: system.moonMesh.material as THREE.Material, baseOpacity: 1 },
    ],
    EARTH_APPROACH,
    planetEaseDwell,
  );
}

export function hideEarthSystem(system: EarthSystem): void {
  hidePlanet(system.root, system.atmosphereMesh);
}
