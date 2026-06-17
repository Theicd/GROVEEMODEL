import * as THREE from "three";

import { fbm, fbm2, lerp, seededRandom } from "./planetProcedural";
import {
  hidePlanet,
  planetAlpha,
  planetEase,
  updatePlanetApproach,
  type PlanetApproachConfig,
} from "./planetApproach";

export const MARS_SIDE = -1;

const MARS_ATMOS_VERTEX = `
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const MARS_ATMOS_FRAGMENT = `
  varying vec3 vNormal;
  void main() {
    float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.5);
    vec3 dustColor = vec3(0.95, 0.5, 0.2);
    vec3 iceColor = vec3(0.7, 0.85, 1.0);
    vec3 color = mix(dustColor, iceColor, 0.15);
    gl_FragColor = vec4(color, 1.0) * intensity * 1.8;
  }
`;

function createMarsColorTexture(): THREE.CanvasTexture {
  const W = 1024;
  const H = 512;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d unavailable");
  const imgData = ctx.createImageData(W, H);
  const d = imgData.data;

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const u = px / W;
      const v = py / H;
      const lat = (v - 0.5) * Math.PI;
      const absLat = Math.abs(lat);
      const n1 = fbm(u * 6 + 0.3, v * 3 + 0.7, 7);
      const n2 = fbm2(u * 12 + 2.1, v * 6 + 4.5, 5) * 0.4;
      const n3 = fbm(u * 24 + 7.0, v * 12 + 3.0, 3) * 0.15;
      const terrain = n1 + n2 + n3;

      let r: number;
      let g: number;
      let b: number;

      if (absLat > 1.2) {
        const polarBlend = Math.min(1, (absLat - 1.2) / 0.25);
        const polarNoise = fbm(u * 20, v * 10, 3) * 0.3;
        r = lerp(210, 240, polarBlend + polarNoise);
        g = lerp(195, 232, polarBlend + polarNoise);
        b = lerp(180, 225, polarBlend + polarNoise);
      } else if (terrain < 0.35) {
        const t = terrain / 0.35;
        r = lerp(110, 150, t);
        g = lerp(55, 75, t);
        b = lerp(30, 45, t);
      } else if (terrain < 0.55) {
        const t = (terrain - 0.35) / 0.2;
        r = lerp(150, 185, t);
        g = lerp(75, 100, t);
        b = lerp(45, 60, t);
      } else if (terrain < 0.75) {
        const t = (terrain - 0.55) / 0.2;
        r = lerp(185, 210, t);
        g = lerp(100, 140, t);
        b = lerp(60, 85, t);
      } else {
        const t = (terrain - 0.75) / 0.25;
        r = lerp(210, 230, t);
        g = lerp(140, 185, t);
        b = lerp(85, 140, t);
      }

      if (absLat > 0.95 && absLat <= 1.2) {
        const fade = (absLat - 0.95) / 0.25;
        r = lerp(r, 220, fade * 0.6);
        g = lerp(g, 200, fade * 0.6);
        b = lerp(b, 185, fade * 0.6);
      }

      const idx = (py * W + px) * 4;
      d[idx] = Math.min(255, Math.max(0, Math.round(r)));
      d[idx + 1] = Math.min(255, Math.max(0, Math.round(g)));
      d[idx + 2] = Math.min(255, Math.max(0, Math.round(b)));
      d[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createMarsBumpTexture(): THREE.CanvasTexture {
  const W = 1024;
  const H = 512;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d unavailable");
  const imgData = ctx.createImageData(W, H);
  const d = imgData.data;

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const u = px / W;
      const v = py / H;
      const n = fbm(u * 10 + 3.3, v * 5 + 7.7, 8);
      const detail = fbm2(u * 30 + 1.1, v * 15 + 2.2, 4) * 0.3;
      const elevation = Math.min(255, Math.max(0, (n + detail) * 255));
      const idx = (py * W + px) * 4;
      d[idx] = d[idx + 1] = d[idx + 2] = Math.round(elevation);
      d[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return new THREE.CanvasTexture(canvas);
}

function createMartianMoonTexture(baseColor: string, craterCount: number, seed: number): THREE.CanvasTexture {
  const W = 512;
  const H = 256;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d unavailable");

  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, W, H);

  const rng = seededRandom(seed);
  for (let i = 0; i < craterCount; i++) {
    const cx = rng() * W;
    const cy = rng() * H;
    const radius = rng() * 18 + 3;
    const depth = rng() * 0.6 + 0.2;
    const craterGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    const darkVal = Math.round(80 - depth * 40);
    const midVal = Math.round(110 - depth * 20);
    craterGrad.addColorStop(0, `rgb(${darkVal}, ${darkVal - 5}, ${darkVal - 10})`);
    craterGrad.addColorStop(0.7, `rgb(${midVal}, ${midVal - 5}, ${midVal - 10})`);
    craterGrad.addColorStop(1, "transparent");
    ctx.fillStyle = craterGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export type MarsSystem = {
  root: THREE.Group;
  marsMesh: THREE.Mesh;
  atmosphereMesh: THREE.Mesh;
  phobosPivot: THREE.Group;
  deimosPivot: THREE.Group;
  dispose: () => void;
};

const MARS_APPROACH: PlanetApproachConfig = {
  zFar: -132,
  zNear: -20,
  xFar: 28,
  xNear: 12,
  yFar: 1.8,
  yNear: 0.1,
  scaleFar: 0.85,
  scaleNear: 5.6,
};

export function createMarsSystem(): MarsSystem {
  const marsColorTex = createMarsColorTexture();
  const marsBumpTex = createMarsBumpTexture();
  const phobosTex = createMartianMoonTexture("#7a6a5a", 80, 42);
  const deimosTex = createMartianMoonTexture("#8a7a6a", 45, 77);

  const root = new THREE.Group();
  root.rotation.z = (25.2 * Math.PI) / 180;

  const marsGeo = new THREE.SphereGeometry(1, 48, 48);
  const marsMat = new THREE.MeshPhongMaterial({
    map: marsColorTex,
    bumpMap: marsBumpTex,
    bumpScale: 0.1,
    shininess: 5,
    specular: new THREE.Color(0x221100),
    fog: false,
    transparent: true,
    opacity: 1,
  });
  const marsMesh = new THREE.Mesh(marsGeo, marsMat);
  root.add(marsMesh);

  const phobosGeo = new THREE.SphereGeometry(0.14, 24, 24);
  const phobosMat = new THREE.MeshPhongMaterial({
    map: phobosTex,
    bumpMap: phobosTex,
    bumpScale: 0.05,
    fog: false,
    transparent: true,
    opacity: 1,
  });
  const phobosMesh = new THREE.Mesh(phobosGeo, phobosMat);
  phobosMesh.scale.set(1.4, 1.1, 0.9);
  const phobosPivot = new THREE.Group();
  phobosPivot.rotation.x = 0.1;
  phobosMesh.position.set(1.55, 0.05, 0);
  phobosPivot.add(phobosMesh);
  root.add(phobosPivot);

  const deimosGeo = new THREE.SphereGeometry(0.09, 24, 24);
  const deimosMat = new THREE.MeshPhongMaterial({
    map: deimosTex,
    bumpMap: deimosTex,
    bumpScale: 0.04,
    fog: false,
    transparent: true,
    opacity: 1,
  });
  const deimosMesh = new THREE.Mesh(deimosGeo, deimosMat);
  deimosMesh.scale.set(1.3, 1.1, 0.85);
  const deimosPivot = new THREE.Group();
  deimosPivot.rotation.x = -0.15;
  deimosPivot.rotation.z = 0.1;
  deimosMesh.position.set(2.35, -0.08, 0);
  deimosPivot.add(deimosMesh);
  root.add(deimosPivot);

  const atmosphereGeo = new THREE.SphereGeometry(1.06, 48, 48);
  const atmosphereMat = new THREE.ShaderMaterial({
    vertexShader: MARS_ATMOS_VERTEX,
    fragmentShader: MARS_ATMOS_FRAGMENT,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true,
    opacity: 1,
    depthWrite: false,
    fog: false,
  });
  const atmosphereMesh = new THREE.Mesh(atmosphereGeo, atmosphereMat);

  const dispose = () => {
    marsGeo.dispose();
    phobosGeo.dispose();
    deimosGeo.dispose();
    atmosphereGeo.dispose();
    marsMat.dispose();
    phobosMat.dispose();
    deimosMat.dispose();
    atmosphereMat.dispose();
    marsColorTex.dispose();
    marsBumpTex.dispose();
    phobosTex.dispose();
    deimosTex.dispose();
  };

  root.visible = false;
  atmosphereMesh.visible = false;

  return { root, marsMesh, atmosphereMesh, phobosPivot, deimosPivot, dispose };
}

export function updateMarsApproach(system: MarsSystem, cycleLinear: number, dt: number, side = MARS_SIDE): number {
  return updatePlanetApproach(
    system.root,
    system.atmosphereMesh,
    cycleLinear,
    dt,
    side,
    [{ mesh: system.marsMesh, speed: 0.07 }],
    [
      { pivot: system.phobosPivot, speed: 0.55 },
      { pivot: system.deimosPivot, speed: 0.22 },
    ],
    [
      { material: system.marsMesh.material as THREE.Material, baseOpacity: 1 },
      {
        material:
          system.phobosPivot.children[0] instanceof THREE.Mesh
            ? (system.phobosPivot.children[0].material as THREE.Material)
            : null,
        baseOpacity: 1,
      },
      {
        material:
          system.deimosPivot.children[0] instanceof THREE.Mesh
            ? (system.deimosPivot.children[0].material as THREE.Material)
            : null,
        baseOpacity: 1,
      },
    ],
    MARS_APPROACH,
  );
}

export function hideMarsSystem(system: MarsSystem): void {
  hidePlanet(system.root, system.atmosphereMesh);
}

export { planetEase, planetAlpha };
