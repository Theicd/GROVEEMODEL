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
