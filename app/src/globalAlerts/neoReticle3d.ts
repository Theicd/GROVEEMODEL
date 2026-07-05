import * as THREE from "three";

/** FPS-style corner brackets (targeting square) in local XY plane. */
export function createCornerReticle(size: number, color: number, opacity = 0.92): THREE.LineSegments {
  const h = size * 0.5;
  const leg = size * 0.38;
  const z = 0;
  const pts: number[] = [];
  const corner = (cx: number, cy: number, dx: number, dy: number) => {
    pts.push(cx, cy, z, cx + dx * leg, cy, z);
    pts.push(cx, cy, z, cx, cy + dy * leg, z);
  };
  corner(-h, h, 1, -1);
  corner(h, h, -1, -1);
  corner(-h, -h, 1, 1);
  corner(h, -h, -1, 1);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
    }),
  );
}

export function createNeoHeadMesh(radius: number, color: number): THREE.Group {
  const g = new THREE.Group();
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 16),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.98 }),
  );
  g.add(core);
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(radius * 2.6, 12, 12),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  g.add(glow);
  const reticle = createCornerReticle(radius * 5.5, color, 0.95);
  g.add(reticle);
  return g;
}

export function billboardToCamera(obj: THREE.Object3D, camera: THREE.Camera) {
  obj.quaternion.copy(camera.quaternion);
}

export type SelectionFrame = {
  group: THREE.Group;
  ring: THREE.LineLoop;
  brackets: THREE.LineSegments;
  bracketMat: THREE.LineBasicMaterial;
  ringMat: THREE.LineBasicMaterial;
};

/** Game-style selection frame: rotating dashed ring + corner brackets, billboarded. */
export function createSelectionFrame(color = 0x66ddff): SelectionFrame {
  const group = new THREE.Group();

  const segs = 64;
  const ringPts: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    ringPts.push(Math.cos(a), Math.sin(a), 0);
  }
  const ringGeo = new THREE.BufferGeometry();
  ringGeo.setAttribute("position", new THREE.Float32BufferAttribute(ringPts, 3));
  const ringMat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
    depthTest: false,
  });
  const ring = new THREE.LineLoop(ringGeo, ringMat);
  group.add(ring);

  const b = 1.28;
  const leg = 0.42;
  const bracketPts: number[] = [];
  const corner = (cx: number, cy: number, dx: number, dy: number) => {
    bracketPts.push(cx, cy, 0, cx + dx * leg, cy, 0);
    bracketPts.push(cx, cy, 0, cx, cy + dy * leg, 0);
  };
  corner(-b, b, 1, -1);
  corner(b, b, -1, -1);
  corner(-b, -b, 1, 1);
  corner(b, -b, -1, 1);
  const bracketGeo = new THREE.BufferGeometry();
  bracketGeo.setAttribute("position", new THREE.Float32BufferAttribute(bracketPts, 3));
  const bracketMat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.95,
    depthWrite: false,
    depthTest: false,
  });
  const brackets = new THREE.LineSegments(bracketGeo, bracketMat);
  group.add(brackets);

  group.visible = false;
  group.renderOrder = 999;
  return { group, ring, brackets, bracketMat, ringMat };
}
