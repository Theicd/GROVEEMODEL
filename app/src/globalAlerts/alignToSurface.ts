import * as THREE from "three";

/** Orient a group so local X/Y lie on the surface tangent plane and +Z points outward. */
export function alignGroupToSurface(group: THREE.Object3D, surfacePos: THREE.Vector3): void {
  const normal = surfacePos.clone().normalize();
  const upRef = new THREE.Vector3(0, 1, 0);
  let tangent = new THREE.Vector3().crossVectors(upRef, normal);
  if (tangent.lengthSq() < 1e-6) {
    tangent.set(1, 0, 0).cross(normal);
  }
  tangent.normalize();
  const bitangent = new THREE.Vector3().crossVectors(normal, tangent).normalize();
  const m = new THREE.Matrix4().makeBasis(tangent, bitangent, normal);
  group.quaternion.setFromRotationMatrix(m);
}

export function latLonToVec3(lat: number, lon: number, r: number): THREE.Vector3 {
  const phi = ((90 - lat) * Math.PI) / 180;
  const theta = ((lon + 180) * Math.PI) / 180;
  return new THREE.Vector3(
    -r * Math.sin(phi) * Math.cos(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.sin(theta),
  );
}
