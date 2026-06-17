import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
  createEarthSystem,
  EARTH_SIDE,
  hideEarthSystem,
  updateEarthApproach,
  type EarthSystem,
} from "./earthBackground";
import { createMarsSystem, hideMarsSystem, MARS_SIDE, updateMarsApproach, type MarsSystem } from "./marsBackground";
import { getSolarJourneyCycleSec, getSolarJourneyState } from "./solarJourney";

const CYAN = 0x00f3ff;
const GREEN = 0x10a37f;
const SPACE_FOG = 0x000000;

/** Slow forward drift through space (units/sec toward camera). */
const WARP_LAYERS = [
  { count: 2200, spread: 110, depth: 95, speed: 0.28, size: 0.022, opacity: 0.42, color: 0x8899aa },
  { count: 1100, spread: 65, depth: 72, speed: 0.55, size: 0.038, opacity: 0.55, color: 0xaabbcc },
  { count: 420, spread: 38, depth: 50, speed: 1.05, size: 0.052, opacity: 0.72, color: 0xcce0ff },
  { count: 90, spread: 50, depth: 85, speed: 0.38, size: 0.11, opacity: 0.92, color: 0xffffff },
] as const;

/** Full fly-by cycle: distant dot → passes overhead (~2 min). */
const FLY_CYCLE_SEC = 120;
const Z_FAR = -62;
const Z_PAST = 5.8;

type ArcLayer = {
  group: THREE.Group;
  speed: number;
};

type HudArcDef = {
  radius: number;
  thickness: number;
  start: number;
  length: number;
  speed: number;
  tiltX: number;
  tiltY: number;
  z: number;
  color: number;
  glow: number;
};

const HUD_ARCS: HudArcDef[] = [
  { radius: 0.42, thickness: 0.022, start: 0.35, length: Math.PI * 1.55, speed: 0.0016, tiltX: 0.08, tiltY: 0, z: 0, color: CYAN, glow: 1 },
  { radius: 0.42, thickness: 0.008, start: Math.PI * 1.05, length: Math.PI * 0.42, speed: 0.0016, tiltX: 0.08, tiltY: 0, z: 0.002, color: GREEN, glow: 0.7 },
  { radius: 0.58, thickness: 0.014, start: Math.PI * 0.15, length: Math.PI * 1.1, speed: -0.0012, tiltX: -0.12, tiltY: 0.18, z: -0.01, color: CYAN, glow: 0.85 },
  { radius: 0.58, thickness: 0.007, start: Math.PI * 1.35, length: Math.PI * 0.55, speed: -0.0012, tiltX: -0.12, tiltY: 0.18, z: -0.008, color: GREEN, glow: 0.55 },
  { radius: 0.74, thickness: 0.011, start: 0.9, length: Math.PI * 0.95, speed: 0.0009, tiltX: 0.15, tiltY: -0.22, z: 0.015, color: CYAN, glow: 0.65 },
  { radius: 0.74, thickness: 0.006, start: Math.PI * 1.65, length: Math.PI * 0.48, speed: 0.0009, tiltX: 0.15, tiltY: -0.22, z: 0.018, color: GREEN, glow: 0.45 },
  { radius: 0.9, thickness: 0.009, start: Math.PI * 0.5, length: Math.PI * 1.25, speed: -0.0007, tiltX: -0.06, tiltY: 0.1, z: -0.02, color: CYAN, glow: 0.5 },
  { radius: 0.9, thickness: 0.005, start: 0.05, length: Math.PI * 0.35, speed: -0.0007, tiltX: -0.06, tiltY: 0.1, z: -0.018, color: GREEN, glow: 0.35 },
];

function makeArcMesh(
  def: HudArcDef,
  scaleGlow: number,
  opacity: number,
  additive: boolean,
): THREE.Mesh {
  const inner = def.radius - def.thickness * scaleGlow * 0.5;
  const outer = def.radius + def.thickness * scaleGlow * 0.5;
  const geo = new THREE.RingGeometry(inner, outer, 96, 1, def.start, def.length);
  const mat = new THREE.MeshBasicMaterial({
    color: def.color,
    transparent: true,
    opacity: opacity * def.glow,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    depthWrite: !additive,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.z = def.z;
  mesh.rotation.x = def.tiltX;
  mesh.rotation.y = def.tiltY;
  return mesh;
}

function buildArcGroup(def: HudArcDef): ArcLayer {
  const group = new THREE.Group();
  group.add(makeArcMesh(def, 1, 0.92, false));
  group.add(makeArcMesh(def, 1.35, 0.38, true));
  group.add(makeArcMesh(def, 2.1, 0.16, true));
  group.add(makeArcMesh(def, 3.2, 0.07, true));
  return { group, speed: def.speed };
}

/** Smooth start, steady cruise, rush past at end — feels like deep-space approach. */
function flyEase(linear: number): number {
  if (linear < 0.12) {
    const t = linear / 0.12;
    return t * t * 0.12;
  }
  if (linear < 0.78) {
    return 0.12 + ((linear - 0.12) / 0.66) * 0.58;
  }
  const t = (linear - 0.78) / 0.22;
  return 0.7 + t * t * 0.3;
}

function flyAlpha(linear: number): number {
  if (linear < 0.06) return linear / 0.06;
  if (linear > 0.9) return Math.max(0, (1 - linear) / 0.1);
  return 1;
}

type WarpStarLayer = {
  points: THREE.Points;
  geo: THREE.BufferGeometry;
  positions: Float32Array;
  count: number;
  spread: number;
  depth: number;
  speed: number;
};

function seedStar(positions: Float32Array, i: number, spread: number, depth: number) {
  positions[i * 3] = (Math.random() - 0.5) * spread;
  positions[i * 3 + 1] = (Math.random() - 0.5) * spread;
  positions[i * 3 + 2] = -5 - Math.random() * depth;
}

function createWarpStarLayer(
  count: number,
  spread: number,
  depth: number,
  speed: number,
  size: number,
  opacity: number,
  color: number,
): WarpStarLayer {
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) seedStar(positions, i, spread, depth);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const points = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      color,
      size,
      transparent: true,
      opacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    }),
  );
  return { points, geo, positions, count, spread, depth, speed };
}

function advanceWarpLayer(layer: WarpStarLayer, dt: number, passZ = 4.5) {
  const { positions, count, spread, depth, speed } = layer;
  for (let i = 0; i < count; i++) {
    positions[i * 3 + 2] += speed * dt;
    if (positions[i * 3 + 2] > passZ) seedStar(positions, i, spread, depth);
  }
  layer.geo.attributes.position.needsUpdate = true;
}

function collectFadeMaterials(root: THREE.Object3D): THREE.Material[] {
  const mats: THREE.Material[] = [];
  root.traverse((obj) => {
    if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments || obj instanceof THREE.Points) {
      const m = obj.material;
      if (Array.isArray(m)) mats.push(...m);
      else mats.push(m);
    }
  });
  return mats;
}

export function GroveeHudCanvas({ contained = false }: { contained?: boolean }) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const measure = () => {
      if (contained) {
        const parent = mount.parentElement;
        if (!parent) return { width: 0, height: 0 };
        return { width: parent.clientWidth, height: parent.clientHeight };
      }
      return { width: window.innerWidth, height: window.innerHeight };
    };

    let { width, height } = measure();
    if (width <= 0 || height <= 0) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(SPACE_FOG);
    scene.fog = new THREE.Fog(SPACE_FOG, 45, 190);

    const camera = new THREE.PerspectiveCamera(38, width / height, 0.05, 220);
    camera.position.set(0, 0.05, 2.85);
    camera.lookAt(0, 0, -20);

    const renderer = new THREE.WebGLRenderer({ alpha: false, antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.setClearColor(SPACE_FOG, 1);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    mount.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0x222233, 0.45);
    scene.add(ambientLight);
    const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
    sunLight.position.set(8, 4, 6);
    scene.add(sunLight);

    const marsFill = new THREE.PointLight(0xffccaa, 0, 28);
    marsFill.position.set(-4, 1.5, 4);
    scene.add(marsFill);

    let earthSystem: EarthSystem | null = null;
    let marsSystem: MarsSystem | null = null;
    let journeyTime = 0;
    let cancelled = false;
    const textureLoader = new THREE.TextureLoader();
    createEarthSystem(textureLoader)
      .then((system) => {
        if (cancelled) {
          system.dispose();
          return;
        }
        earthSystem = system;
        scene.add(system.root);
        scene.add(system.atmosphereMesh);
      })
      .catch(() => {
        /* textures optional — stars + HUD still render */
      });

    try {
      marsSystem = createMarsSystem();
      scene.add(marsSystem.root);
      scene.add(marsSystem.atmosphereMesh);
    } catch {
      marsSystem = null;
    }

    const warpStars: WarpStarLayer[] = WARP_LAYERS.map((cfg) =>
      createWarpStarLayer(cfg.count, cfg.spread, cfg.depth, cfg.speed, cfg.size, cfg.opacity, cfg.color),
    );
    for (const layer of warpStars) scene.add(layer.points);

    const craft = new THREE.Group();
    scene.add(craft);

    const layers: ArcLayer[] = HUD_ARCS.map(buildArcGroup);
    for (const layer of layers) craft.add(layer.group);

    const coreGeo = new THREE.RingGeometry(0.1, 0.128, 64, 1, 0, Math.PI * 2);
    const coreMat = new THREE.MeshBasicMaterial({
      color: CYAN,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    craft.add(core);

    const coreGlowGeo = new THREE.RingGeometry(0.06, 0.16, 64, 1, 0, Math.PI * 2);
    const coreGlowMat = new THREE.MeshBasicMaterial({
      color: GREEN,
      transparent: true,
      opacity: 0.2,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const coreGlow = new THREE.Mesh(coreGlowGeo, coreGlowMat);
    craft.add(coreGlow);

    const tickGeo = new THREE.BufferGeometry();
    const tickCount = 48;
    const tickPos = new Float32Array(tickCount * 3 * 2);
    for (let i = 0; i < tickCount; i++) {
      const a = (i / tickCount) * Math.PI * 2;
      const r0 = 0.96;
      const r1 = i % 4 === 0 ? 1.02 : 0.99;
      tickPos[i * 6] = Math.cos(a) * r0;
      tickPos[i * 6 + 1] = Math.sin(a) * r0;
      tickPos[i * 6 + 2] = 0.03;
      tickPos[i * 6 + 3] = Math.cos(a) * r1;
      tickPos[i * 6 + 4] = Math.sin(a) * r1;
      tickPos[i * 6 + 5] = 0.03;
    }
    tickGeo.setAttribute("position", new THREE.BufferAttribute(tickPos, 3));
    const tickMat = new THREE.LineBasicMaterial({ color: CYAN, transparent: true, opacity: 0.22 });
    const ticks = new THREE.LineSegments(tickGeo, tickMat);
    craft.add(ticks);

    const dustGeo = new THREE.BufferGeometry();
    const dustCount = 100;
    const dustPos = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      const r = 0.35 + Math.random() * 1.1;
      const a = Math.random() * Math.PI * 2;
      dustPos[i * 3] = Math.cos(a) * r;
      dustPos[i * 3 + 1] = Math.sin(a) * r;
      dustPos[i * 3 + 2] = (Math.random() - 0.5) * 0.08;
    }
    dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
    const dustMat = new THREE.PointsMaterial({
      color: CYAN,
      size: 0.012,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const dust = new THREE.Points(dustGeo, dustMat);
    craft.add(dust);

    const fadeMaterials = collectFadeMaterials(craft);
    const baseOpacities = new Map<THREE.Material, number>();
    for (const mat of fadeMaterials) {
      if ("opacity" in mat && typeof mat.opacity === "number") {
        baseOpacities.set(mat, mat.opacity);
      }
    }

    const clock = new THREE.Clock();
    let cycleTime = reducedMotion ? FLY_CYCLE_SEC * 0.45 : 0;
    let raf = 0;
    let spinT = 0;
    const tempColor = new THREE.Color();

    const animate = () => {
      const dt = Math.min(clock.getDelta(), 0.05);
      cycleTime = (cycleTime + dt) % FLY_CYCLE_SEC;
      spinT += dt;
      journeyTime = (journeyTime + dt) % getSolarJourneyCycleSec();
      const journey = getSolarJourneyState(journeyTime);

      const linear = cycleTime / FLY_CYCLE_SEC;
      const eased = flyEase(linear);
      const alpha = flyAlpha(linear);

      if (reducedMotion) {
        craft.position.z = -8;
        craft.rotation.set(0.04, 0.1, 0);
        for (const layer of warpStars) advanceWarpLayer(layer, dt * 0.35);
      } else {
        craft.position.z = Z_FAR + (Z_PAST - Z_FAR) * eased;
        craft.position.x = Math.sin(eased * Math.PI * 2) * 0.08 * (1 - eased);
        craft.position.y = Math.cos(eased * Math.PI * 1.5) * 0.05 * (1 - eased);
        craft.rotation.y = eased * 0.42;
        craft.rotation.x = Math.sin(eased * Math.PI) * 0.14;
        craft.rotation.z = Math.sin(spinT * 0.15) * 0.04;

        for (const layer of warpStars) advanceWarpLayer(layer, dt);
        camera.position.z = 2.85 + Math.sin(spinT * 0.08) * 0.015;
      }

      const pulse = 0.5 + 0.5 * Math.sin(spinT * 1.8);
      tempColor.setHex(CYAN).lerp(new THREE.Color(GREEN), pulse * 0.35);
      coreMat.color.copy(tempColor);
      coreMat.opacity = (0.45 + pulse * 0.25) * alpha;
      coreGlowMat.opacity = 0.2 * alpha;
      coreGlow.scale.setScalar(1 + pulse * 0.08);
      core.rotation.z += dt * 0.35;
      tickMat.opacity = 0.22 * alpha;
      dustMat.opacity = 0.45 * alpha;

      for (const layer of layers) {
        layer.group.rotation.z += layer.speed * (1 + eased * 2.5);
      }
      ticks.rotation.z -= dt * 0.025;
      dust.rotation.z += dt * 0.015;

      for (const mat of fadeMaterials) {
        const base = baseOpacities.get(mat);
        if (base !== undefined && "opacity" in mat) {
          mat.opacity = base * alpha;
        }
      }

      if (earthSystem) {
        if (journey.planetId === "earth") {
          updateEarthApproach(earthSystem, journey.linear, dt, EARTH_SIDE);
        } else {
          hideEarthSystem(earthSystem);
        }
      }

      if (marsSystem) {
        if (journey.planetId === "mars") {
          updateMarsApproach(marsSystem, journey.linear, dt, MARS_SIDE);
          marsFill.intensity = 1.2;
        } else {
          hideMarsSystem(marsSystem);
          marsFill.intensity = 0;
        }
      }

      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const resize = () => {
      const next = measure();
      if (next.width <= 0 || next.height <= 0) return;
      width = next.width;
      height = next.height;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    let ro: ResizeObserver | undefined;
    if (contained) {
      const parent = mount.parentElement;
      if (parent) {
        ro = new ResizeObserver(resize);
        ro.observe(parent);
      }
    } else {
      window.addEventListener("resize", resize);
    }

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      ro?.disconnect();
      if (!contained) window.removeEventListener("resize", resize);

      const disposeObj = (obj: THREE.Object3D) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments || obj instanceof THREE.Points) {
          obj.geometry.dispose();
          const mat = obj.material;
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else mat.dispose();
        }
      };

      craft.traverse(disposeObj);
      for (const layer of warpStars) disposeObj(layer.points);
      if (earthSystem) {
        scene.remove(earthSystem.root);
        scene.remove(earthSystem.atmosphereMesh);
        earthSystem.dispose();
      }
      if (marsSystem) {
        scene.remove(marsSystem.root);
        scene.remove(marsSystem.atmosphereMesh);
        marsSystem.dispose();
      }
      marsFill.dispose();
      ambientLight.dispose();
      sunLight.dispose();
      coreGeo.dispose();
      coreGlowGeo.dispose();
      tickGeo.dispose();
      dustGeo.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [contained]);

  return <div ref={mountRef} className="grovee-hud-canvas bg-canvas" aria-hidden="true" />;
}
