import * as THREE from "three";

/** Smooth fly-by easing — slow start, cruise, rush at end. */
export function planetEase(linear: number): number {
  if (linear < 0.08) {
    const t = linear / 0.08;
    return t * t * 0.08;
  }
  if (linear < 0.82) {
    return 0.08 + ((linear - 0.08) / 0.74) * 0.72;
  }
  const t = (linear - 0.82) / 0.18;
  return 0.8 + t * t * 0.2;
}

export function planetAlpha(linear: number): number {
  if (linear < 0.04) return linear / 0.04;
  if (linear > 0.96) return Math.max(0, (1 - linear) / 0.04);
  return 1;
}

export type PlanetApproachConfig = {
  zFar?: number;
  zNear?: number;
  xFar?: number;
  xNear?: number;
  yFar?: number;
  yNear?: number;
  scaleFar?: number;
  scaleNear?: number;
};

const DEFAULT_APPROACH: Required<PlanetApproachConfig> = {
  zFar: -128,
  zNear: -18,
  xFar: 26,
  xNear: 11,
  yFar: 2.2,
  yNear: 0.15,
  scaleFar: 1.1,
  scaleNear: 6.2,
};

export function updatePlanetApproach(
  root: THREE.Group,
  atmosphereMesh: THREE.Mesh | null,
  cycleLinear: number,
  dt: number,
  side: number,
  spinMeshes: Array<{ mesh: THREE.Mesh; speed: number }>,
  moonPivots: Array<{ pivot: THREE.Group; speed: number }>,
  opacityMaterials: Array<{ material: THREE.Material | null; baseOpacity: number }>,
  config: PlanetApproachConfig = {},
): number {
  const cfg = { ...DEFAULT_APPROACH, ...config };
  const eased = planetEase(cycleLinear);
  const alpha = planetAlpha(cycleLinear);

  const z = THREE.MathUtils.lerp(cfg.zFar, cfg.zNear, eased);
  const x = side * THREE.MathUtils.lerp(cfg.xFar, cfg.xNear, eased);
  const y = THREE.MathUtils.lerp(cfg.yFar, cfg.yNear, eased);
  const scale = THREE.MathUtils.lerp(cfg.scaleFar, cfg.scaleNear, eased);

  root.position.set(x, y, z);
  root.scale.setScalar(scale);
  root.visible = alpha > 0.02;

  if (atmosphereMesh) {
    atmosphereMesh.position.copy(root.position);
    atmosphereMesh.scale.copy(root.scale);
    atmosphereMesh.visible = root.visible;
    if (atmosphereMesh.material instanceof THREE.Material && "opacity" in atmosphereMesh.material) {
      atmosphereMesh.material.opacity = alpha;
    }
  }

  if (root.visible) {
    for (const { mesh, speed } of spinMeshes) {
      mesh.rotation.y += dt * speed;
    }
    for (const { pivot, speed } of moonPivots) {
      pivot.rotation.y += dt * speed;
    }
  }

  for (const { material, baseOpacity } of opacityMaterials) {
    if (material && "opacity" in material && typeof material.opacity === "number") {
      material.opacity = baseOpacity * alpha;
    }
  }

  return alpha;
}

export function hidePlanet(
  root: THREE.Group,
  atmosphereMesh: THREE.Mesh | null,
): void {
  root.visible = false;
  if (atmosphereMesh) atmosphereMesh.visible = false;
}
