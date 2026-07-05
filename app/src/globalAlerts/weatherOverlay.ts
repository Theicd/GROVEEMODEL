import * as THREE from "three";
import { latLonToVec3 } from "./alignToSurface";
import type { WeatherCell, WeatherCellKind } from "./weatherGrid";

const WEATHER_R = 1.024;

const KIND_COLOR: Record<Exclude<WeatherCellKind, "clear">, number> = {
  cloudy: 0xccccdd,
  rain: 0x3399ff,
  snow: 0xaaddff,
  thunder: 0xffee55,
  fog: 0x888899,
};

const KIND_SIZE: Record<Exclude<WeatherCellKind, "clear">, number> = {
  cloudy: 0.018,
  rain: 0.014,
  snow: 0.013,
  thunder: 0.022,
  fog: 0.02,
};

export type WeatherOverlayHandle = {
  sync: (cells: WeatherCell[]) => void;
  update: (time: number) => void;
  setVisible: (visible: boolean) => void;
  dispose: () => void;
};

export function createWeatherOverlay(parent: THREE.Object3D): WeatherOverlayHandle {
  const group = new THREE.Group();
  parent.add(group);

  const thunderGroups: THREE.Group[] = [];
  let cloudPulseMesh: THREE.Points | null = null;

  function sync(cells: WeatherCell[]) {
    while (group.children.length) {
      const ch = group.children[0];
      group.remove(ch);
      ch.traverse((node) => {
        if (node instanceof THREE.Mesh || node instanceof THREE.Points) {
          node.geometry?.dispose();
          const m = node.material;
          if (Array.isArray(m)) m.forEach((mat) => mat.dispose());
          else m?.dispose();
        }
      });
    }
    thunderGroups.length = 0;
    cloudPulseMesh = null;

    const byKind = new Map<Exclude<WeatherCellKind, "clear">, WeatherCell[]>();
    for (const c of cells) {
      if (c.kind === "clear") continue;
      const list = byKind.get(c.kind) ?? [];
      list.push(c);
      byKind.set(c.kind, list);
    }

    for (const [kind, list] of byKind) {
      if (kind === "thunder") {
        for (const cell of list) {
          const g = new THREE.Group();
          const pos = latLonToVec3(cell.lat, cell.lon, WEATHER_R);
          g.position.copy(pos);

          const core = new THREE.Mesh(
            new THREE.SphereGeometry(0.01, 8, 8),
            new THREE.MeshBasicMaterial({
              color: KIND_COLOR.thunder,
              transparent: true,
              opacity: 0.85,
              blending: THREE.AdditiveBlending,
              depthWrite: false,
            }),
          );
          g.add(core);

          const bolt = new THREE.Mesh(
            new THREE.PlaneGeometry(0.035, 0.055),
            new THREE.MeshBasicMaterial({
              color: 0xffffff,
              transparent: true,
              opacity: 0,
              blending: THREE.AdditiveBlending,
              depthWrite: false,
              side: THREE.DoubleSide,
            }),
          );
          bolt.position.z = 0.008;
          g.add(bolt);

          (g.userData as { bolt: THREE.Mesh; phase: number }).bolt = bolt;
          (g.userData as { bolt: THREE.Mesh; phase: number }).phase = Math.random() * Math.PI * 2;

          group.add(g);
          thunderGroups.push(g);
        }
        continue;
      }

      const positions: number[] = [];
      const colors: number[] = [];
      const color = new THREE.Color(KIND_COLOR[kind]);
      const sz = KIND_SIZE[kind];

      for (const cell of list) {
        const p = latLonToVec3(cell.lat, cell.lon, WEATHER_R);
        positions.push(p.x, p.y, p.z);
        colors.push(color.r, color.g, color.b);
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

      const pts = new THREE.Points(
        geo,
        new THREE.PointsMaterial({
          size: sz,
          vertexColors: true,
          transparent: true,
          opacity: kind === "cloudy" ? 0.35 : 0.72,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          sizeAttenuation: true,
        }),
      );
      group.add(pts);
      if (kind === "cloudy") cloudPulseMesh = pts;
    }
  }

  function update(time: number) {
    if (cloudPulseMesh?.material instanceof THREE.PointsMaterial) {
      cloudPulseMesh.material.opacity = 0.28 + 0.12 * Math.sin(time * 0.55);
    }

    for (const g of thunderGroups) {
      const ud = g.userData as { bolt: THREE.Mesh; phase: number };
      const flash = Math.max(0, Math.sin(time * 9 + ud.phase));
      const strike = flash > 0.92 ? 1 : flash > 0.85 ? 0.35 : 0;
      (ud.bolt.material as THREE.MeshBasicMaterial).opacity = strike;
      const core = g.children[0] as THREE.Mesh;
      (core.material as THREE.MeshBasicMaterial).opacity = 0.45 + strike * 0.55;
    }
  }

  function dispose() {
    group.traverse((ch) => {
      if (ch instanceof THREE.Mesh || ch instanceof THREE.Points) {
        ch.geometry?.dispose();
        const m = ch.material;
        if (Array.isArray(m)) m.forEach((mat) => mat.dispose());
        else m?.dispose();
      }
    });
    parent.remove(group);
  }

  function setVisible(visible: boolean) {
    group.visible = visible;
  }

  return { sync, update, setVisible, dispose };
}
