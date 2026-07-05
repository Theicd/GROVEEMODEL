import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { alignGroupToSurface, latLonToVec3 } from "./alignToSurface";
import { neoSpacePosition, visualRadiusFromDistAu, visualRadiusFromLd, MOON_SCENE_ORBIT, MOON_SCENE_RADIUS, type NeoOrbitTrack } from "./neoTrack";
import { EQ_LIVE_WINDOW_MS } from "./types";
import { buildApproachDisplayTrack } from "./neoApproachTrack";
import { estimateLiveDistLd, interpolateNeoPoint } from "./neoLiveMetrics";
import { createNeoLabelSprite, drawNeoLabel, type NeoLabelSprite } from "./neoLabelSprite";
import { billboardToCamera, createCornerReticle, createNeoHeadMesh, createSelectionFrame, type SelectionFrame } from "./neoReticle3d";
import { type StormTrack, type StormTrackPoint, stormPositionNow } from "./parseStormGeometry";
import { fetchGlobalWeatherCells } from "./fetchGlobalWeatherCells";
import { getHurricaneIntensity, parseWindKmh } from "./hurricaneIntensity";
import { createWeatherOverlay } from "./weatherOverlay";
import {
  createAsteroidGeometry,
  createAsteroidTexture,
  createCometTail,
  inferNeoVisualProfile,
  orbitLineColor,
  orientCometTail,
  spaceDisplayMeshSize,
  SPECTRAL_TYPES,
} from "./spaceObjectVisuals";
import { EVENT_TYPE_LABELS, type GlobeAlertEvent, type GlobeAlertEventType } from "./types";

type ActiveEffect = {
  group: THREE.Group;
  mat?: THREE.ShaderMaterial;
  pts?: THREE.Points;
  em?: THREE.MeshBasicMaterial;
  eye?: THREE.Mesh;
  age: number;
  maxAge: number;
  type: GlobeAlertEventType;
  eventId: string;
  spinSpeed?: number;
  eyePulseHz?: number;
};

type MarkerEntry = {
  group: THREE.Group;
  type: GlobeAlertEventType;
  ringMat: THREE.ShaderMaterial;
  coreMat: THREE.MeshBasicMaterial;
  magnitude?: number;
};

const EARTH_ROT = 0.001;
const ASTEROID_SPIN_DISPLAY = 420;
const EQ_RED = 0xff2200;
const INITIAL_CAMERA_DIST = 7.4;
const SPACE_CAMERA_DIST = 16;
const MIN_ZOOM_DIST = 0.78;
const SPACE_MIN_ZOOM_DIST = 0.45;
const SPACE_MAX_ZOOM_DIST = 52;
const FOCUS_CAMERA_DIST = 1.85;
const STORM_TRACK_R = 1.007;
const STORM_PURPLE = 0xaa66ff;
const STORM_FORECAST = 0xcc99ff;
const NEO_CYAN = 0x66ddff;
const NEO_WARN = 0xffcc44;
const FIREBALL_ORANGE = 0xffaa33;
const MOON_ORBIT_R = MOON_SCENE_ORBIT;
const MOON_TEXTURE_URL =
  "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg";

function eventSeed(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (Math.imul(31, h) + id.charCodeAt(i)) | 0;
  return Math.abs(h);
}

type NeoReticleEntry = {
  group: THREE.Group;
  label: NeoLabelSprite;
  approachLine?: THREE.Line;
  orbitLine?: THREE.Line;
  orbitMat?: THREE.LineBasicMaterial;
  mesh?: THREE.Mesh;
  cometTail?: THREE.Group;
  frame?: THREE.LineSegments;
  frameMat?: THREE.LineBasicMaterial;
  rotAxis?: THREE.Vector3;
  rotSpeed?: number;
  visualSize?: number;
  spaceMode?: boolean;
};

function trackPointsToLine(
  points: StormTrackPoint[],
  color: number,
  opacity: number,
  dashed = false,
): THREE.Line {
  const verts = points.map((p) => latLonToVec3(p.lat, p.lon, STORM_TRACK_R));
  const geo = new THREE.BufferGeometry().setFromPoints(verts);
  const mat = dashed
    ? new THREE.LineDashedMaterial({
        color,
        transparent: true,
        opacity,
        dashSize: 0.018,
        gapSize: 0.01,
      })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(geo, mat);
  if (dashed) line.computeLineDistances();
  return line;
}

function neoTrackToLine(
  points: Array<{ lat: number; lon: number; distLd: number }>,
  color: number,
  opacity: number,
  dashed = false,
  dashStore?: THREE.LineDashedMaterial[],
): THREE.Line {
  const verts = points.map((p) => neoSpacePosition(p.lat, p.lon, p.distLd));
  const geo = new THREE.BufferGeometry().setFromPoints(verts);
  const mat = dashed
    ? new THREE.LineDashedMaterial({
        color,
        transparent: true,
        opacity,
        dashSize: 0.028,
        gapSize: 0.016,
        linewidth: 1,
      })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(geo, mat);
  if (dashed) {
    line.computeLineDistances();
    if (dashStore && mat instanceof THREE.LineDashedMaterial) dashStore.push(mat);
  }
  return line;
}

function disposeObject3D(obj: THREE.Object3D): void {
  obj.traverse((ch) => {
    if (ch instanceof THREE.Mesh || ch instanceof THREE.Line || ch instanceof THREE.Points) {
      ch.geometry?.dispose();
      const m = ch.material;
      if (Array.isArray(m)) m.forEach((mat) => mat.dispose());
      else m?.dispose();
    }
    if (ch instanceof THREE.ArrowHelper) {
      ch.line?.geometry?.dispose();
      ch.cone?.geometry?.dispose();
      (ch.line?.material as THREE.Material)?.dispose();
      (ch.cone?.material as THREE.Material)?.dispose();
    }
  });
}

const RING_VERTEX = `varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;

const RING_FRAGMENT = `
  uniform float uTime; uniform vec3 uColor; varying vec2 vUv;
  void main(){
    vec2 c=vUv-.5; float d=length(c)*2.;
    float pulse=.55+.45*sin(uTime*3.5);
    float wave=fract(d*1.2-uTime*.45);
    float ring=smoothstep(.92,.82,d)*smoothstep(.5,.62,d);
    ring+=smoothstep(.08,.0,abs(wave-.5))*(1.-d)*.7;
    float a=clamp(ring*pulse,0.,1.);
    gl_FragColor=vec4(uColor*(1.+pulse*.8),a);
  }`;

const EQ_BURST_FRAGMENT = `
  uniform float uTime,uInt;uniform vec3 uColor;varying vec2 vUv;
  void main(){
    vec2 c=vUv-.5;float d=length(c)*2.;
    float rings=0.;
    for(int i=0;i<6;i++){
      float ph=float(i)/6.;
      float r=fract(uTime*.28+ph);
      float w=.03+r*.018;
      rings+=smoothstep(w,0.,abs(d-r))*(1.-r)*uInt;
    }
    float pulse=exp(-d*d*28.)*(.6+.4*sin(uTime*12.))*uInt;
    float a=clamp((rings*.85+pulse*1.8)*smoothstep(1.,.25,d),0.,1.);
    gl_FragColor=vec4(uColor*(1.+pulse*2.5),a);
  }`;

function eventColor(ev: GlobeAlertEvent): THREE.Color {
  if (ev.type === "earthquake" || (ev.type === "tsunami" && ev.source === "usgs")) {
    return new THREE.Color(EQ_RED);
  }
  if (ev.type === "tsunami") return new THREE.Color(0x00ccff);
  if (ev.type === "hurricane") {
    const int = getHurricaneIntensity(ev.category, parseWindKmh(ev.severityText));
    return new THREE.Color(int.color);
  }
  if (ev.type === "neo") {
    const ld = ev.distLd ?? 99;
    if (ld < 1) return new THREE.Color(0xff6644);
    if (ld < 5) return new THREE.Color(NEO_WARN);
    return new THREE.Color(NEO_CYAN);
  }
  if (ev.type === "fireball") return new THREE.Color(FIREBALL_ORANGE);
  return new THREE.Color(EVENT_TYPE_LABELS[ev.type].hex);
}

function createPersistentMarker(parent: THREE.Object3D, ev: GlobeAlertEvent): MarkerEntry {
  const isEq = ev.type === "earthquake";
  const isHurricane = ev.type === "hurricane";
  const isNeo = ev.type === "neo";
  const mag = ev.magnitude ?? 4;
  const diamKm = ev.diameterKm ?? 0.05;
  const hurInt = isHurricane
    ? getHurricaneIntensity(ev.category, parseWindKmh(ev.severityText))
    : null;
  const color = eventColor(ev);
  const coreR = isEq
    ? 0.007 + mag * 0.002
    : isNeo
      ? Math.max(0.006, Math.min(0.02, 0.005 + Math.log10(diamKm + 0.01) * 0.006))
      : hurInt?.coreR ?? 0.009;
  const ringOuter = isEq
    ? 0.05 + mag * 0.014
    : isNeo
      ? 0.038 + Math.max(0, 0.05 - (ev.distLd ?? 10) * 0.004)
      : hurInt?.ringOuter ?? 0.042;

  const pos = latLonToVec3(ev.lat, ev.lon, 1.005);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);

  const coreMat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 1,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const core = new THREE.Mesh(new THREE.SphereGeometry(coreR, 14, 14), coreMat);
  g.add(core);

  const ringMat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uColor: { value: color.clone() } },
    vertexShader: RING_VERTEX,
    fragmentShader: RING_FRAGMENT,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  g.add(new THREE.Mesh(new THREE.RingGeometry(ringOuter * 0.35, ringOuter, 40), ringMat));

  if (isEq) {
    g.add(new THREE.PointLight(EQ_RED, 0.35 + mag * 0.12, 0.55));
    const burstMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color(EQ_RED) },
        uInt: { value: Math.min(mag / 3.5, 2.8) },
      },
      vertexShader: RING_VERTEX,
      fragmentShader: EQ_BURST_FRAGMENT,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const burstSz = 0.07 + mag * 0.022;
    g.add(new THREE.Mesh(new THREE.PlaneGeometry(burstSz * 2, burstSz * 2), burstMat));
    (g as THREE.Group & { burstMat?: THREE.ShaderMaterial }).burstMat = burstMat;
  }

  g.userData.eventId = ev.id;
  const pickR = isHurricane
    ? (hurInt?.pickR ?? Math.max(0.09, (ev.category ?? 2) * 0.028))
    : isNeo
      ? 0.065
      : Math.max(ringOuter * 1.35, isEq ? 0.032 + mag * 0.005 : 0.028);
  const pickMesh = new THREE.Mesh(
    new THREE.SphereGeometry(pickR, 10, 10),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
  );
  pickMesh.userData.eventId = ev.id;
  g.add(pickMesh);

  parent.add(g);
  return { group: g, type: ev.type, ringMat, coreMat, magnitude: ev.magnitude };
}

function fxEarthquakeBurst(parent: THREE.Object3D, lat: number, lon: number, mag: number) {
  const pos = latLonToVec3(lat, lon, 1.009);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);
  const sz = 0.1 + mag * 0.028;
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(EQ_RED) },
      uInt: { value: Math.min(mag / 3, 3) },
    },
    vertexShader: RING_VERTEX,
    fragmentShader: EQ_BURST_FRAGMENT,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  g.add(new THREE.Mesh(new THREE.PlaneGeometry(sz * 2, sz * 2), mat));
  g.add(new THREE.PointLight(EQ_RED, mag * 0.55, 0.45));
  parent.add(g);
  return { group: g, mat, age: 0, maxAge: 14 + mag * 0.4, type: "earthquake" as const, eventId: "" };
}

function fxTsunami(parent: THREE.Object3D, lat: number, lon: number, intensity: number) {
  const pos = latLonToVec3(lat, lon, 1.007);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);
  const sz = 0.2 + intensity * 0.07;
  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color(0x00ccff) } },
    vertexShader: RING_VERTEX,
    fragmentShader: `
      uniform float uTime;uniform vec3 uColor;varying vec2 vUv;
      void main(){
        vec2 c=vUv-.5;float d=length(c)*2.;
        float wave=sin(d*22.-uTime*4.5)*.5+.5;
        float env=smoothstep(.95,.7,d)*smoothstep(.1,.25,d);
        float a=wave*env*.75; gl_FragColor=vec4(uColor,a);
      }`,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  g.add(new THREE.Mesh(new THREE.PlaneGeometry(sz * 2, sz * 2), mat));
  parent.add(g);
  return { group: g, mat, age: 0, maxAge: 99999, type: "tsunami" as const, eventId: "" };
}

function fxHurricane(parent: THREE.Object3D, lat: number, lon: number, cat: number, severityText?: string) {
  const int = getHurricaneIntensity(cat, parseWindKmh(severityText));
  const pos = latLonToVec3(lat, lon, 1.012);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);
  const N = int.particleCount;
  const p = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = 0.008 + Math.random() * int.maxRadius;
    p[i * 3] = r * Math.cos(a);
    p[i * 3 + 1] = r * Math.sin(a);
    p[i * 3 + 2] = (Math.random() - 0.5) * 0.006;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(p, 3));
  const pts = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: int.particleSize,
      color: int.color,
      transparent: true,
      opacity: int.opacity,
      depthWrite: false,
      sizeAttenuation: true,
      blending: THREE.AdditiveBlending,
    }),
  );
  g.add(pts);
  const em = new THREE.MeshBasicMaterial({
    color: int.colorSecondary,
    transparent: true,
    opacity: 0.65 + int.category * 0.05,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
  const eye = new THREE.Mesh(
    new THREE.RingGeometry(int.eyeInner, int.eyeOuter, 32),
    em,
  );
  g.add(eye);
  g.add(new THREE.PointLight(int.color, 0.25 + int.category * 0.12, 0.35 + int.category * 0.08));
  parent.add(g);
  return {
    group: g,
    pts,
    eye,
    em,
    age: 0,
    maxAge: 99999,
    type: "hurricane" as const,
    eventId: "",
    spinSpeed: int.spinSpeed,
    eyePulseHz: int.eyePulseHz,
  };
}

function fxFire(parent: THREE.Object3D, lat: number, lon: number, sev: number) {
  const pos = latLonToVec3(lat, lon, 1.006);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);
  const N = 260;
  const pa = new Float32Array(N * 3);
  const cl = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.random() * 0.02 * sev;
    pa[i * 3] = r * Math.cos(a);
    pa[i * 3 + 1] = r * Math.sin(a);
    pa[i * 3 + 2] = Math.random() * 0.05;
    cl[i * 3] = 1;
    cl[i * 3 + 1] = 0.35;
    cl[i * 3 + 2] = 0.05;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pa, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(cl, 3));
  const pts = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: 0.008,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    }),
  );
  g.add(pts);
  g.add(new THREE.PointLight(0xff6600, sev * 0.45, 0.35));
  parent.add(g);
  return { group: g, pts, age: 0, maxAge: 99999, type: "fire" as const, eventId: "" };
}

function fxFlood(parent: THREE.Object3D, lat: number, lon: number, sev: number) {
  const pos = latLonToVec3(lat, lon, 1.007);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);
  const sz = 0.11 + sev * 0.04;
  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color(0x3388ff) } },
    vertexShader: RING_VERTEX,
    fragmentShader: `
      uniform float uTime;uniform vec3 uColor;varying vec2 vUv;
      void main(){
        vec2 c=vUv-.5;float d=length(c)*2.;
        float sp=smoothstep(.15,.85,d)*smoothstep(.95,.75,d);
        float pulse=.65+.35*sin(uTime*2.5+d*8.);
        float a=sp*pulse*.6; gl_FragColor=vec4(uColor*1.4,a);
      }`,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  g.add(new THREE.Mesh(new THREE.PlaneGeometry(sz * 2, sz * 2), mat));
  parent.add(g);
  return { group: g, mat, age: 0, maxAge: 99999, type: "flood" as const, eventId: "" };
}

function fxFireball(parent: THREE.Object3D, ev: GlobeAlertEvent) {
  const pos = latLonToVec3(ev.lat, ev.lon, 1.012);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);
  const kt = ev.impactKt ?? 0.5;
  const sz = 0.08 + Math.min(kt, 5) * 0.02;
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(FIREBALL_ORANGE) },
      uInt: { value: Math.min(kt / 2, 3) },
    },
    vertexShader: RING_VERTEX,
    fragmentShader: EQ_BURST_FRAGMENT,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  g.add(new THREE.Mesh(new THREE.PlaneGeometry(sz * 2.2, sz * 2.2), mat));
  g.add(new THREE.PointLight(FIREBALL_ORANGE, 0.4 + kt * 0.15, 0.35));
  parent.add(g);
  return { group: g, mat, age: 0, maxAge: 99999, type: "fireball" as const, eventId: "" };
}

function fxNeo(parent: THREE.Object3D, ev: GlobeAlertEvent) {
  const pos = latLonToVec3(ev.lat, ev.lon, 1.01);
  const g = new THREE.Group();
  g.position.copy(pos);
  alignGroupToSurface(g, pos);
  const ld = ev.distLd ?? 10;
  const N = 24;
  const p = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const r = 0.012 + Math.min(ld, 20) * 0.0008;
    p[i * 3] = r * Math.cos(a);
    p[i * 3 + 1] = r * Math.sin(a);
    p[i * 3 + 2] = (Math.random() - 0.5) * 0.004;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(p, 3));
  const col = ld < 1 ? 0xff6644 : ld < 5 ? NEO_WARN : NEO_CYAN;
  const pts = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: 0.014,
      color: col,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  g.add(pts);
  parent.add(g);
  return {
    group: g,
    pts,
    age: 0,
    maxAge: 99999,
    type: "neo" as const,
    eventId: "",
    spinSpeed: 0.8 + Math.max(0, 6 - ld) * 0.15,
  };
}

function createAmbientEffect(parent: THREE.Object3D, ev: GlobeAlertEvent): ActiveEffect | null {
  switch (ev.type) {
    case "earthquake":
      return fxEarthquakeBurst(parent, ev.lat, ev.lon, ev.magnitude ?? 4);
    case "tsunami":
      return fxTsunami(parent, ev.lat, ev.lon, ev.severity ?? 1);
    case "hurricane":
      return fxHurricane(parent, ev.lat, ev.lon, ev.category ?? 2, ev.severityText);
    case "fire":
    case "volcano":
      return fxFire(parent, ev.lat, ev.lon, ev.severity ?? 1.2);
    case "flood":
    case "disaster":
      return fxFlood(parent, ev.lat, ev.lon, ev.severity ?? 1);
    case "fireball":
      return fxFireball(parent, ev);
    case "neo":
      return fxNeo(parent, ev);
    default:
      return fxFlood(parent, ev.lat, ev.lon, 0.8);
  }
}

export type GlobeSceneCallbacks = {
  onLoaded?: () => void;
  onFocus?: (ev: GlobeAlertEvent) => void;
  onEventPick?: (ev: GlobeAlertEvent) => void;
};

export type GlobeSceneHandle = {
  syncEvents: (events: GlobeAlertEvent[]) => void;
  focusEvent: (ev: GlobeAlertEvent) => void;
  returnToNormal: () => void;
  showStormTrack: (track: StormTrack) => void;
  clearStormTrack: () => void;
  showNeoTrack: (track: NeoOrbitTrack, diameterKm?: number) => void;
  clearNeoTrack: () => void;
  focusNeoEarthFrame: (track: NeoOrbitTrack) => void;
  focusSpaceNeo: (ev: GlobeAlertEvent) => void;
  clearSpaceNeoFocus: () => void;
  flyToLatLon: (lat: number, lon: number, dist?: number) => void;
  setSpaceMode: (enabled: boolean) => void;
  setNeoOrbitTracks: (tracks: Record<string, NeoOrbitTrack>) => void;
  dispose: () => void;
};

export function initGlobeScene(
  container: HTMLElement,
  callbacks: GlobeSceneCallbacks = {},
): GlobeSceneHandle {
  let width = container.clientWidth;
  let height = container.clientHeight;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(0, 0.12, INITIAL_CAMERA_DIST);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.45;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0x8899bb, 0.85));
  scene.add(new THREE.HemisphereLight(0xaaccff, 0x223344, 0.65));
  const sun = new THREE.DirectionalLight(0xffffff, 3.2);
  sun.position.set(5, 2, 4);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x6688cc, 1.1);
  fill.position.set(-4, -1, -3);
  scene.add(fill);

  const starGeo = new THREE.BufferGeometry();
  const starPos = new Float32Array(4000 * 3);
  for (let i = 0; i < starPos.length; i++) starPos[i] = (Math.random() - 0.5) * 80;
  starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
  const starMesh = new THREE.Points(
    starGeo,
    new THREE.PointsMaterial({ size: 0.05, color: 0xffffff, transparent: true, opacity: 0.85 }),
  );
  scene.add(starMesh);

  const earthGroup = new THREE.Group();
  earthGroup.rotation.z = (23.5 * Math.PI) / 180;
  scene.add(earthGroup);

  const texLoader = new THREE.TextureLoader();
  const earthMat = new THREE.MeshPhongMaterial({
    color: 0x4488cc,
    emissive: 0x112244,
    emissiveIntensity: 0.35,
    shininess: 18,
    specular: new THREE.Color(0x444466),
  });
  const earthMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), earthMat);
  earthGroup.add(earthMesh);

  const cloudsMesh = new THREE.Mesh(
    new THREE.SphereGeometry(1.018, 64, 64),
    new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.5, color: 0xffffff }),
  );
  earthGroup.add(cloudsMesh);

  const weatherOverlay = createWeatherOverlay(earthGroup);
  let weatherRefreshTimer = 0;

  async function refreshWeatherLayer() {
    try {
      const cells = await fetchGlobalWeatherCells();
      weatherOverlay.sync(cells);
    } catch {
      /* non-fatal */
    }
  }

  void refreshWeatherLayer();

  const atmosphereMesh = new THREE.Mesh(
    new THREE.SphereGeometry(1.22, 64, 64),
    new THREE.ShaderMaterial({
      vertexShader: `varying vec3 vN;void main(){vN=normalize(normalMatrix*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
      fragmentShader: `varying vec3 vN;void main(){float i=pow(.65-dot(vN,vec3(0,0,1.)),3.5);gl_FragColor=vec4(.35,.65,1.,1.)*i*1.2;}`,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    }),
  );
  scene.add(atmosphereMesh);

  texLoader.load(
    "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
    (map) => {
      map.colorSpace = THREE.SRGBColorSpace;
      earthMat.map = map;
      earthMat.emissiveIntensity = 0.12;
      earthMat.needsUpdate = true;
    },
  );
  texLoader.load("https://unpkg.com/three-globe/example/img/earth-topology.png", (bump) => {
    earthMat.bumpMap = bump;
    earthMat.bumpScale = 0.04;
    earthMat.needsUpdate = true;
  });
  texLoader.load("https://unpkg.com/three-globe/example/img/earth-water.png", (spec) => {
    earthMat.specularMap = spec;
    earthMat.needsUpdate = true;
  });
  texLoader.load(
    "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_clouds_1024.png",
    (map) => {
      (cloudsMesh.material as THREE.MeshLambertMaterial).map = map;
      (cloudsMesh.material as THREE.MeshLambertMaterial).needsUpdate = true;
      callbacks.onLoaded?.();
    },
    undefined,
    () => callbacks.onLoaded?.(),
  );

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = MIN_ZOOM_DIST;
  controls.maxDistance = 22;
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.target.set(0, 0, 0);

  let userControlActive = false;
  let spaceCameraReleased = false;
  controls.addEventListener("start", () => {
    userControlActive = true;
    if (spaceMode) {
      eventLockActive = false;
      spaceCameraReleased = true;
    }
  });
  controls.addEventListener("end", () => {
    userControlActive = false;
  });

  const activeEffects: ActiveEffect[] = [];
  const activeEvents: GlobeAlertEvent[] = [];
  const eventsById = new Map<string, GlobeAlertEvent>();
  const markers = new Map<string, MarkerEntry>();
  const stormTrackGroup = new THREE.Group();
  earthGroup.add(stormTrackGroup);
  const neoTrackGroup = new THREE.Group();
  scene.add(neoTrackGroup);
  const neoReticleGroup = new THREE.Group();
  scene.add(neoReticleGroup);
  const neoReticles = new Map<string, NeoReticleEntry>();
  const neoDashMats: THREE.LineDashedMaterial[] = [];

  const selectionFrame: SelectionFrame = createSelectionFrame(0x66ddff);
  scene.add(selectionFrame.group);

  const moonOrbitG = new THREE.Group();
  moonOrbitG.visible = false;
  scene.add(moonOrbitG);
  const moonPivot = new THREE.Group();
  moonPivot.position.set(MOON_ORBIT_R, 0, 0);
  moonOrbitG.add(moonPivot);
  const moonGeo = new THREE.SphereGeometry(MOON_SCENE_RADIUS, 48, 48);
  const moonMat = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    shininess: 8,
    specular: new THREE.Color(0x333333),
  });
  const moonMesh = new THREE.Mesh(moonGeo, moonMat);
  moonMesh.rotation.z = (6.68 * Math.PI) / 180;
  moonPivot.add(moonMesh);
  let moonMap: THREE.Texture | null = null;
  texLoader.load(
    MOON_TEXTURE_URL,
    (map) => {
      map.colorSpace = THREE.SRGBColorSpace;
      moonMap = map;
      moonMat.map = map;
      moonMat.needsUpdate = true;
    },
  );
  const moonOrbitPts: THREE.Vector3[] = [];
  for (let i = 0; i <= 64; i++) {
    const t = (i / 64) * Math.PI * 2;
    moonOrbitPts.push(new THREE.Vector3(MOON_ORBIT_R * Math.cos(t), 0, MOON_ORBIT_R * Math.sin(t)));
  }
  const moonOrbitLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(moonOrbitPts),
    new THREE.LineBasicMaterial({ color: 0x445566, transparent: true, opacity: 0.14, depthWrite: false }),
  );
  moonOrbitLine.visible = false;
  scene.add(moonOrbitLine);

  let spaceMode = false;
  let neoOrbitTracks: Record<string, NeoOrbitTrack> = {};

  function lerpTrackPoint(a: NeoOrbitTrack["points"][0], b: NeoOrbitTrack["points"][0], u: number) {
    return {
      t: a.t + (b.t - a.t) * u,
      lat: a.lat + (b.lat - a.lat) * u,
      lon: a.lon + (b.lon - a.lon) * u,
      distAu: a.distAu + (b.distAu - a.distAu) * u,
      distLd: a.distLd + (b.distLd - a.distLd) * u,
      deldotKmS: a.deldotKmS + (b.deldotKmS - a.deldotKmS) * u,
    };
  }

  function pointAlongClosedTrack(track: NeoOrbitTrack, phase: number) {
    const pts = track.points;
    if (pts.length < 2) return pts[0] ?? track.closest;
    const f = (phase % 1) * pts.length;
    const idx = Math.floor(f) % pts.length;
    const next = (idx + 1) % pts.length;
    return lerpTrackPoint(pts[idx], pts[next], f - Math.floor(f));
  }

  function resolveNeoWorldPosition(ev: GlobeAlertEvent): THREE.Vector3 {
    const liveLd = ev.showcaseNeo ? (ev.distLd ?? 10) : estimateLiveDistLd(ev);
    const track = neoOrbitTracks[ev.id];
    if (track && track.points.length >= 2) {
      if (ev.showcaseNeo) {
        const phase = (globalTime / 180) % 1;
        const p = pointAlongClosedTrack(track, phase);
        return neoSpacePosition(p.lat, p.lon, p.distLd);
      }
      const p = interpolateNeoPoint(track, Date.now());
      return neoSpacePosition(p.lat, p.lon, p.distLd);
    }
    return neoSpacePosition(ev.lat, ev.lon, liveLd);
  }

  function applyOrbitFromTrack(entry: NeoReticleEntry, eventId: string) {
    const track = neoOrbitTracks[eventId];
    const ev = eventsById.get(eventId);
    if (entry.approachLine) entry.approachLine.visible = false;

    if (!spaceMode || !entry.orbitLine || !track || track.points.length < 2 || !ev) {
      if (entry.orbitLine) entry.orbitLine.visible = false;
      return;
    }

    const pts = track.points.map((p) => neoSpacePosition(p.lat, p.lon, p.distLd));
    if (ev.showcaseNeo && pts.length > 2) pts.push(pts[0].clone());
    entry.orbitLine.geometry.dispose();
    entry.orbitLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    entry.orbitLine.visible = true;
  }
  let activeStormTrack: StormTrack | null = null;
  let stormHeadMesh: THREE.Mesh | null = null;
  let activeNeoTrack: NeoOrbitTrack | null = null;
  let activeNeoDisplay: NeoOrbitTrack | null = null;
  let neoHeadMesh: THREE.Group | null = null;
  let neoApproachLine: THREE.Line | null = null;
  let neoTrailPts: THREE.Points | null = null;
  let eventLockActive = false;
  let focusEvent: GlobeAlertEvent | null = null;
  let spaceFocusId: string | null = null;
  let shakeIntensity = 0;
  let disposed = false;
  let raf = 0;
  const clock = new THREE.Timer();
  clock.connect(document);
  let globalTime = 0;

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: 0.08 };
  const pointerNdc = new THREE.Vector2();
  let pointerDown: { x: number; y: number } | null = null;

  function resolveEventIdFromHit(obj: THREE.Object3D | null): string | undefined {
    let cur: THREE.Object3D | null = obj;
    while (cur) {
      if (typeof cur.userData.eventId === "string") return cur.userData.eventId;
      cur = cur.parent;
    }
    return undefined;
  }

  function pickEventAt(clientX: number, clientY: number): GlobeAlertEvent | null {
    const rect = renderer.domElement.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
      return null;
    }
    pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointerNdc, camera);
    const roots: THREE.Object3D[] = [];
    markers.forEach((m) => roots.push(m.group));
    neoReticles.forEach((g) => roots.push(g.group));
    const hits = raycaster.intersectObjects(roots, true);
    if (!hits.length) return null;
    const eventId = resolveEventIdFromHit(hits[0].object);
    if (!eventId) return null;
    return eventsById.get(eventId) ?? null;
  }

  function disposeNeoReticleEntry(entry: NeoReticleEntry) {
    neoReticleGroup.remove(entry.group);
    if (entry.approachLine) neoReticleGroup.remove(entry.approachLine);
    if (entry.orbitLine) neoReticleGroup.remove(entry.orbitLine);
    entry.group.traverse((ch) => {
      if (ch instanceof THREE.LineSegments || ch instanceof THREE.Mesh || ch instanceof THREE.Points) {
        ch.geometry?.dispose();
        const mat = ch.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat?.dispose();
      }
    });
    if (entry.approachLine) {
      entry.approachLine.geometry?.dispose();
      (entry.approachLine.material as THREE.Material)?.dispose();
    }
    if (entry.orbitLine) {
      entry.orbitLine.geometry?.dispose();
      (entry.orbitLine.material as THREE.Material)?.dispose();
    }
    (entry.label.sprite.material as THREE.SpriteMaterial).map?.dispose();
    (entry.label.sprite.material as THREE.Material)?.dispose();
  }

  function clearNeoReticles() {
    for (const entry of neoReticles.values()) {
      disposeNeoReticleEntry(entry);
    }
    neoReticles.clear();
  }

  function updateNeoReticleLabels() {
    for (const [id, entry] of neoReticles) {
      const ev = eventsById.get(id);
      if (!ev || ev.type !== "neo") continue;
      const liveLd = ev.showcaseNeo ? (ev.distLd ?? liveEstShowcase(ev)) : estimateLiveDistLd(ev);
      const speed = ev.vRel ?? ev.vInf ?? 0;
      const color = ev.isPha || liveLd < 1 ? "#ff5555" : liveLd < 5 ? "#ffcc44" : "#66ddff";
      drawNeoLabel(entry.label, {
        title: ev.location.slice(0, 22),
        dist: `${liveLd.toFixed(2)} LD`,
        speed: `${speed.toFixed(1)} km/s`,
        eta: encounterLabel(ev),
      }, color);
      entry.label.sprite.visible = spaceMode;
      const pos = resolveNeoWorldPosition(ev);
      entry.group.position.copy(pos);
    }
  }

  function liveEstShowcase(ev: GlobeAlertEvent): number {
    return ev.distLd ?? 10;
  }

  /** Hours (or days) until the object's close encounter with Earth. */
  function encounterLabel(ev: GlobeAlertEvent): string {
    if (ev.showcaseNeo) return "מחזורי";
    const ms = (ev.approachTime ?? ev.time) - Date.now();
    if (ms <= 0) return "מפגש חלף";
    const h = ms / 3_600_000;
    if (h < 48) return `מפגש ${Math.round(h)}ש'`;
    return `מפגש ${Math.round(h / 24)}י'`;
  }

  function syncNeoReticles(events: GlobeAlertEvent[]) {
    const want = new Set(events.filter((e) => e.type === "neo").map((e) => e.id));
    for (const [id, entry] of neoReticles) {
      if (!want.has(id)) {
        disposeNeoReticleEntry(entry);
        neoReticles.delete(id);
      }
    }
    for (const ev of events) {
      if (ev.type !== "neo") continue;
      const liveLd = estimateLiveDistLd(ev);
      const pos = resolveNeoWorldPosition(ev);
      const color = liveLd < 5 ? NEO_WARN : NEO_CYAN;
      const size = 0.18 + Math.min(0.08, liveLd * 0.006);
      let entry = neoReticles.get(ev.id);
      if (entry && entry.spaceMode !== spaceMode) {
        disposeNeoReticleEntry(entry);
        neoReticles.delete(ev.id);
        entry = undefined;
      }
      if (!entry) {
        const group = new THREE.Group();
        const label = createNeoLabelSprite();
        group.add(label.sprite);

        if (spaceMode) {
          const profile = inferNeoVisualProfile(ev, eventSeed(ev.id));
          const spec = SPECTRAL_TYPES[profile.spectral];
          const meshSize = spaceDisplayMeshSize(profile.visualSize, liveLd);
          const geo = createAsteroidGeometry(profile.shape, meshSize, eventSeed(ev.id));
          const tex = createAsteroidTexture(profile.spectral, eventSeed(ev.id));
          const mesh = new THREE.Mesh(
            geo,
            new THREE.MeshStandardMaterial({
              color: spec.color,
              roughness: spec.roughness,
              metalness: spec.metalness,
              map: tex.map,
              bumpMap: tex.bumpMap,
              bumpScale: meshSize * 0.25,
            }),
          );
          mesh.userData.eventId = ev.id;
          group.add(mesh);
          let cometTail: THREE.Group | undefined;
          if (profile.spectral === "comet") {
            cometTail = createCometTail(group, meshSize);
          }

          const frameColor = orbitLineColor(profile.spectral, !!ev.isPha, liveLd);
          const frame = createCornerReticle(meshSize * 4.2, frameColor, 0.5);
          frame.userData.eventId = ev.id;
          frame.renderOrder = 20;
          group.add(frame);

          label.sprite.visible = true;
          label.sprite.position.set(0, meshSize * 3.4, 0);

          const orbitMat = new THREE.LineBasicMaterial({
            color: frameColor,
            transparent: true,
            opacity: 0.22,
            depthWrite: false,
          });
          const orbitLine = new THREE.Line(new THREE.BufferGeometry(), orbitMat);
          orbitLine.renderOrder = 5;
          neoReticleGroup.add(orbitLine);

          neoReticleGroup.add(group);
          entry = {
            group,
            label,
            mesh,
            cometTail,
            frame,
            frameMat: frame.material as THREE.LineBasicMaterial,
            orbitLine,
            orbitMat,
            rotAxis: profile.rotAxis,
            rotSpeed: profile.rotSpeed,
            visualSize: meshSize,
            spaceMode: true,
          };
        } else {
          group.add(createCornerReticle(size, color));
          const core = new THREE.Mesh(
            new THREE.SphereGeometry(0.018, 12, 12),
            new THREE.MeshBasicMaterial({
              color: 0xffffff,
              transparent: true,
              opacity: 0.95,
              blending: THREE.AdditiveBlending,
              depthWrite: false,
            }),
          );
          group.add(core);
          label.sprite.position.set(0, size * 1.8, 0);
          const approachMat = new THREE.LineDashedMaterial({
            color,
            transparent: true,
            opacity: 0.45,
            dashSize: 0.06,
            gapSize: 0.035,
            depthWrite: false,
          });
          const approachLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), pos.clone()]),
            approachMat,
          );
          approachLine.computeLineDistances();
          neoReticleGroup.add(approachLine);
          neoReticleGroup.add(group);
          entry = { group, label, approachLine, spaceMode: false };
          neoDashMats.push(approachMat);
        }
        neoReticles.set(ev.id, entry);
      }
      const posNow = resolveNeoWorldPosition(ev);
      entry.group.position.copy(posNow);
      entry.group.userData.eventId = ev.id;
      applyOrbitFromTrack(entry, ev.id);
    }
    updateNeoReticleLabels();
  }

  function onPointerDown(e: PointerEvent) {
    pointerDown = { x: e.clientX, y: e.clientY };
  }

  function onPointerUp(e: PointerEvent) {
    if (!pointerDown) return;
    const dx = e.clientX - pointerDown.x;
    const dy = e.clientY - pointerDown.y;
    pointerDown = null;
    if (dx * dx + dy * dy > 100) return;

    const ev = pickEventAt(e.clientX, e.clientY);
    if (!ev) return;
    focusOnEvent(ev);
    callbacks.onEventPick?.(ev);
  }

  renderer.domElement.style.touchAction = "none";
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);

  function updateEffects(dt: number) {
    globalTime += dt;
    for (const entry of markers.values()) {
      entry.ringMat.uniforms.uTime.value = globalTime;
      const pulse = 0.85 + 0.15 * Math.sin(globalTime * 4 + (entry.magnitude ?? 1));
      entry.coreMat.opacity = pulse;
      const ext = entry.group as THREE.Group & { burstMat?: THREE.ShaderMaterial };
      if (ext.burstMat) ext.burstMat.uniforms.uTime.value = globalTime;
    }

    for (let i = activeEffects.length - 1; i >= 0; i--) {
      const fx = activeEffects[i];
      fx.age += dt;
      if (fx.maxAge < 90000 && fx.age >= fx.maxAge) {
        earthMesh.remove(fx.group);
        fx.group.traverse((ch) => {
          if (ch instanceof THREE.Mesh || ch instanceof THREE.Points) {
            ch.geometry?.dispose();
            if (Array.isArray(ch.material)) ch.material.forEach((m) => m.dispose());
            else ch.material?.dispose();
          }
        });
        activeEffects.splice(i, 1);
        continue;
      }
      if (fx.mat?.uniforms?.uTime) fx.mat.uniforms.uTime.value = globalTime;

      if (fx.type === "hurricane" && fx.pts) {
        const spin = fx.spinSpeed ?? 0.35;
        const pa = fx.pts.geometry.attributes.position.array as Float32Array;
        const n = pa.length / 3;
        for (let j = 0; j < n; j++) {
          const x = pa[j * 3];
          const y = pa[j * 3 + 1];
          const r = Math.max(0.001, Math.sqrt(x * x + y * y));
          const a = Math.atan2(y, x);
          pa[j * 3] = r * Math.cos(a + (dt * spin) / (r * 3.5 + 0.08));
          pa[j * 3 + 1] = r * Math.sin(a + (dt * spin) / (r * 3.5 + 0.08));
        }
        fx.pts.geometry.attributes.position.needsUpdate = true;
        if (fx.em && fx.eye) {
          const hz = fx.eyePulseHz ?? 1.5;
          const amp = 0.12 + (spin - 0.2) * 0.12;
          const p = 1 + Math.sin(globalTime * hz) * amp;
          fx.eye.scale.set(p, p, p);
          fx.em.opacity = 0.55 + Math.sin(globalTime * hz * 1.3) * 0.15;
        }
      }
      if (fx.type === "neo" && fx.pts && fx.spinSpeed) {
        fx.group.rotation.z += dt * fx.spinSpeed;
      }
      if (fx.type === "fire" && fx.pts) {
        const pa = fx.pts.geometry.attributes.position.array as Float32Array;
        const ca = fx.pts.geometry.attributes.color.array as Float32Array;
        for (let j = 0; j < pa.length / 3; j++) {
          pa[j * 3 + 2] += dt * 0.028;
          pa[j * 3] += (Math.random() - 0.5) * dt * 0.014;
          pa[j * 3 + 1] += (Math.random() - 0.5) * dt * 0.014;
          const t = Math.max(0, pa[j * 3 + 2] / 0.05);
          ca[j * 3] = 1;
          ca[j * 3 + 1] = Math.max(0, 0.35 + (1 - t) * 0.55);
          ca[j * 3 + 2] = 0.05;
          if (pa[j * 3 + 2] > 0.05) {
            const a = Math.random() * Math.PI * 2;
            const r = Math.random() * 0.018;
            pa[j * 3] = r * Math.cos(a);
            pa[j * 3 + 1] = r * Math.sin(a);
            pa[j * 3 + 2] = 0;
          }
        }
        fx.pts.geometry.attributes.position.needsUpdate = true;
        fx.pts.geometry.attributes.color.needsUpdate = true;
      }
    }
  }

  function syncEvents(events: GlobeAlertEvent[]) {
    const wantIds = new Set(events.map((e) => e.id));

    for (const ev of events) {
      eventsById.set(ev.id, ev);
      const idx = activeEvents.findIndex((e) => e.id === ev.id);
      if (idx >= 0) activeEvents[idx] = ev;
      else activeEvents.unshift(ev);

      if (ev.type === "neo") continue;

      const skipMarker = ev.trackPending || (ev.lat === 0 && ev.lon === 0);
      const existing = markers.get(ev.id);

      if (skipMarker) {
        if (existing) {
          earthMesh.remove(existing.group);
          existing.group.traverse((ch) => {
            if (ch instanceof THREE.Mesh || ch instanceof THREE.Points) {
              ch.geometry?.dispose();
              if (Array.isArray(ch.material)) ch.material.forEach((mat) => mat.dispose());
              else ch.material?.dispose();
            }
          });
          markers.delete(ev.id);
        }
        continue;
      }

      if (!existing) {
        markers.set(ev.id, createPersistentMarker(earthMesh, ev));
        const fx = createAmbientEffect(earthMesh, ev);
        if (fx) {
          fx.eventId = ev.id;
          activeEffects.push(fx);
          if (ev.type === "earthquake" && (ev.magnitude ?? 0) >= 4.5) {
            shakeIntensity = Math.max(shakeIntensity, (ev.magnitude ?? 4.5) * 0.002);
          }
        }
      }
    }

    const now = Date.now();
    for (let i = activeEvents.length - 1; i >= 0; i--) {
      const ev = activeEvents[i];
      const dropped = !wantIds.has(ev.id);
      const neoExpired =
        ev.type === "neo" && (ev.approachTime ?? ev.time) + 3 * 86400000 < now;
      const fireballExpired = ev.type === "fireball" && now - ev.time > 90 * 86400000;
      const eqExpired =
        ev.source === "usgs" &&
        (ev.type === "earthquake" || ev.type === "tsunami") &&
        now - ev.time > EQ_LIVE_WINDOW_MS;
      const gdacsExpired =
        ev.source === "gdacs" &&
        ((ev.gdacsEndTime != null && ev.gdacsEndTime < now) || ev.gdacsIsCurrent === false);
      const defaultExpired =
        ev.type !== "neo" &&
        ev.type !== "fireball" &&
        ev.type !== "earthquake" &&
        ev.type !== "tsunami" &&
        ev.source !== "gdacs" &&
        now - ev.time > EQ_LIVE_WINDOW_MS;
      if (!dropped && !neoExpired && !fireballExpired && !eqExpired && !gdacsExpired && !defaultExpired)
        continue;

      const id = activeEvents[i].id;
      const m = markers.get(id);
      if (m) {
        earthMesh.remove(m.group);
        m.group.traverse((ch) => {
          if (ch instanceof THREE.Mesh || ch instanceof THREE.Points) {
            ch.geometry?.dispose();
            if (Array.isArray(ch.material)) ch.material.forEach((mat) => mat.dispose());
            else ch.material?.dispose();
          }
        });
        markers.delete(id);
      }
      activeEvents.splice(i, 1);
    }
    if (activeEvents.length > 80) activeEvents.length = 80;
    syncNeoReticles(events);
  }

  function clearStormTrack() {
    disposeObject3D(stormTrackGroup);
    stormTrackGroup.clear();
    activeStormTrack = null;
    stormHeadMesh = null;
  }

  function showStormTrack(track: StormTrack) {
    clearStormTrack();
    activeStormTrack = track;
    eventLockActive = true;

    if (track.observed.length >= 2) {
      stormTrackGroup.add(trackPointsToLine(track.observed, STORM_PURPLE, 0.9));
    }

    const forecastLine: StormTrackPoint[] =
      track.observed.length > 0
        ? [{ ...track.observed[track.observed.length - 1], kind: "forecast" }, ...track.forecast]
        : [...track.forecast];
    if (forecastLine.length >= 2) {
      stormTrackGroup.add(trackPointsToLine(forecastLine, STORM_FORECAST, 0.72, true));
    }

    for (const fp of track.forecast) {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.0045, 8, 8),
        new THREE.MeshBasicMaterial({ color: STORM_FORECAST, transparent: true, opacity: 0.85 }),
      );
      dot.position.copy(latLonToVec3(fp.lat, fp.lon, STORM_TRACK_R));
      stormTrackGroup.add(dot);
    }

    const cur = stormPositionNow(track);
    const next = track.forecast[0];
    if (next) {
      const a = latLonToVec3(cur.lat, cur.lon, STORM_TRACK_R);
      const b = latLonToVec3(next.lat, next.lon, STORM_TRACK_R);
      const dir = b.clone().sub(a);
      const len = dir.length();
      if (len > 1e-4) {
        stormTrackGroup.add(new THREE.ArrowHelper(dir.normalize(), a, Math.min(len * 0.9, 0.12), STORM_PURPLE, 0.022, 0.014));
      }
    }

    const head = stormPositionNow(track);
    stormHeadMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.009, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 }),
    );
    stormHeadMesh.position.copy(latLonToVec3(head.lat, head.lon, STORM_TRACK_R + 0.003));
    stormTrackGroup.add(stormHeadMesh);
  }

  function clearNeoTrack() {
    disposeObject3D(neoTrackGroup);
    neoTrackGroup.clear();
    neoDashMats.length = 0;
    activeNeoTrack = null;
    activeNeoDisplay = null;
    neoHeadMesh = null;
    neoApproachLine = null;
    neoTrailPts = null;
  }

  function frameEarthAndNeoTrack(track: NeoOrbitTrack) {
    if (!track.points.length) {
      const c = track.closest;
      const neoWp = neoSpacePosition(c.lat, c.lon, c.distLd);
      const center = neoWp.clone().lerp(new THREE.Vector3(0, 0, 0), 0.35);
      controls.target.copy(center);
      const dist = 2.2 + visualRadiusFromDistAu(c.distAu) * 1.2;
      camera.position.copy(center.clone().add(new THREE.Vector3(0.3, 0.22, 1).normalize().multiplyScalar(dist)));
      return;
    }
    const box = new THREE.Box3();
    box.expandByPoint(new THREE.Vector3(0, 0, 0));
    for (const p of track.points) {
      box.expandByPoint(neoSpacePosition(p.lat, p.lon, p.distLd));
    }
    box.expandByScalar(1.15);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y, size.z, 2.8);
    controls.target.copy(center);
    const offset = new THREE.Vector3(0.28, 0.22, 1).normalize().multiplyScalar(span * 0.95 + 1.4);
    camera.position.copy(center.clone().add(offset));
  }

  let activeNeoDiameterKm = 0.05;

  function showNeoTrack(track: NeoOrbitTrack, diameterKm?: number) {
    if (spaceMode) return;
    clearNeoTrack();
    clearStormTrack();
    clearNeoReticles();
    activeNeoTrack = track;
    activeNeoDisplay = buildApproachDisplayTrack(track);
    activeNeoDiameterKm = diameterKm ?? 0.05;
    eventLockActive = true;

    const display = activeNeoDisplay;
    if (display.points.length >= 2) {
      neoTrackGroup.add(neoTrackToLine(display.points, 0x224466, 0.5, true, neoDashMats));
      neoTrackGroup.add(neoTrackToLine(display.points, NEO_CYAN, 1, true, neoDashMats));
      neoTrackGroup.add(neoTrackToLine(display.points, 0xffffff, 0.25, true, neoDashMats));
    }

    for (let i = 0; i < display.points.length; i++) {
      const p = display.points[i];
      const wp = neoSpacePosition(p.lat, p.lon, p.distLd);
      const t = i / Math.max(1, display.points.length - 1);
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.006 + t * 0.012, 8, 8),
        new THREE.MeshBasicMaterial({
          color: NEO_CYAN,
          transparent: true,
          opacity: 0.35 + t * 0.55,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        }),
      );
      dot.position.copy(wp);
      neoTrackGroup.add(dot);
    }

    const c = display.closest;
    const missR = Math.max(0.02, 0.014 + c.distLd * 0.012);
    const missRing = new THREE.Mesh(
      new THREE.RingGeometry(missR * 0.45, missR, 48),
      new THREE.MeshBasicMaterial({
        color: c.distLd < 1 ? 0xff4444 : c.distLd < 5 ? NEO_WARN : NEO_CYAN,
        transparent: true,
        opacity: 0.75,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    const ringG = new THREE.Group();
    ringG.position.copy(latLonToVec3(c.lat, c.lon, 1.006));
    alignGroupToSurface(ringG, ringG.position);
    ringG.add(missRing);
    neoTrackGroup.add(ringG);

    const posNow = interpolateNeoPoint(display);
    const headPos = neoSpacePosition(posNow.lat, posNow.lon, posNow.distLd);
    const headColor = posNow.distLd < 1 ? 0xff6644 : posNow.distLd < 5 ? NEO_WARN : 0xffffff;
    const headR = Math.max(0.022, Math.min(0.055, 0.018 + Math.log10(activeNeoDiameterKm + 0.01) * 0.012));
    neoHeadMesh = createNeoHeadMesh(headR, headColor);
    neoHeadMesh.position.copy(headPos);
    neoTrackGroup.add(neoHeadMesh);

    const approachMat = new THREE.LineDashedMaterial({
      color: NEO_WARN,
      transparent: true,
      opacity: 0.55,
      dashSize: 0.08,
      gapSize: 0.04,
      depthWrite: false,
    });
    const approachLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), headPos.clone()]),
      approachMat,
    );
    approachLine.computeLineDistances();
    neoTrackGroup.add(approachLine);
    neoApproachLine = approachLine;
    neoDashMats.push(approachMat);

    const trailN = 32;
    const trailPos = new Float32Array(trailN * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    neoTrailPts = new THREE.Points(
      trailGeo,
      new THREE.PointsMaterial({
        size: 0.022,
        color: NEO_WARN,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
      }),
    );
    neoTrackGroup.add(neoTrailPts);

    frameEarthAndNeoTrack(track);
  }

  function applyNeoFocusHighlight() {
    for (const [id, entry] of neoReticles) {
      if (!entry.mesh) continue;
      const mat = entry.mesh.material as THREE.MeshStandardMaterial;
      const isFocus = spaceMode && spaceFocusId === id;
      mat.emissive.setHex(isFocus ? 0x335577 : 0x000000);
      mat.emissiveIntensity = isFocus ? 0.45 : 0;
      entry.group.scale.setScalar(isFocus ? 1.12 : 1);
    }
    if (!spaceMode || !spaceFocusId || !neoReticles.has(spaceFocusId)) {
      selectionFrame.group.visible = false;
    }
  }

  const _selWp = new THREE.Vector3();
  function updateSelectionFrame() {
    if (!spaceMode || !spaceFocusId) {
      selectionFrame.group.visible = false;
      return;
    }
    const entry = neoReticles.get(spaceFocusId);
    const ev = eventsById.get(spaceFocusId);
    if (!entry || !ev) {
      selectionFrame.group.visible = false;
      return;
    }
    entry.group.getWorldPosition(_selWp);
    selectionFrame.group.position.copy(_selWp);
    billboardToCamera(selectionFrame.group, camera);

    const meshR = (entry.visualSize ?? 0.05) * (entry.group.scale.x || 1);
    const camDist = camera.position.distanceTo(_selWp);
    // Keep the frame comfortably larger than the body, scaled a bit with camera distance.
    const baseR = Math.max(meshR * 2.6, camDist * 0.045);
    const pulse = 1 + 0.05 * Math.sin(globalTime * 3.5);
    selectionFrame.group.scale.setScalar(baseR * pulse);
    selectionFrame.ring.rotation.z += 0.01;
    const flick = 0.75 + 0.25 * Math.sin(globalTime * 4);
    selectionFrame.bracketMat.opacity = flick;
    selectionFrame.ringMat.opacity = 0.45 + 0.2 * Math.sin(globalTime * 2.2 + 1);
    selectionFrame.group.visible = true;
  }

  /** Close framing distance so the focused body appears large in view. */
  function focusCamDistFor(entry: NeoReticleEntry): number {
    const meshR = (entry.visualSize ?? 0.05) * (entry.group.scale.x || 1);
    return Math.max(0.32, meshR * 7);
  }

  function updateSpaceFocusCamera() {
    if (!spaceMode || !spaceFocusId || userControlActive || spaceCameraReleased || !eventLockActive) return;
    const entry = neoReticles.get(spaceFocusId);
    const ev = eventsById.get(spaceFocusId);
    if (!entry || !ev) return;
    const wp = new THREE.Vector3();
    entry.group.getWorldPosition(wp);
    const camDist = focusCamDistFor(entry);
    const frameCenter = wp.clone();
    controls.target.lerp(frameCenter, 0.12);
    const viewDir = wp.clone().sub(controls.target).normalize();
    if (viewDir.lengthSq() < 1e-6) viewDir.copy(camera.position).sub(wp).normalize();
    const side = new THREE.Vector3().crossVectors(viewDir, new THREE.Vector3(0, 1, 0));
    if (side.lengthSq() < 1e-6) side.set(1, 0, 0);
    side.normalize();
    const desired = controls.target
      .clone()
      .add(viewDir.multiplyScalar(camDist))
      .add(side.multiplyScalar(camDist * 0.22));
    camera.position.lerp(desired, 0.08);
  }

  function focusSpaceNeo(ev: GlobeAlertEvent) {
    if (!spaceMode || ev.type !== "neo") return;
    clearStormTrack();
    clearNeoTrack();
    focusEvent = ev;
    spaceFocusId = ev.id;
    eventLockActive = true;
    spaceCameraReleased = false;
    userControlActive = false;
    const liveLd = ev.showcaseNeo ? (ev.distLd ?? 10) : estimateLiveDistLd(ev);
    const frameColor = ev.showcaseNeo
      ? 0x66aaff
      : ev.isPha || liveLd < 1
        ? 0xff4d4d
        : liveLd < 5
          ? 0xffcc44
          : 0x66ddff;
    selectionFrame.bracketMat.color.setHex(frameColor);
    selectionFrame.ringMat.color.setHex(frameColor);
    applyNeoFocusHighlight();
    callbacks.onFocus?.(ev);
  }

  function clearSpaceNeoFocus() {
    spaceFocusId = null;
    if (focusEvent?.type === "neo") focusEvent = null;
    eventLockActive = false;
    applyNeoFocusHighlight();
  }

  function focusNeoEarthFrame(track: NeoOrbitTrack) {
    if (userControlActive) return;
    eventLockActive = true;
    frameEarthAndNeoTrack(track);
  }

  function flyToLatLon(lat: number, lon: number, dist = FOCUS_CAMERA_DIST) {
    if (userControlActive) return;
    const wp = latLonToVec3(lat, lon, STORM_TRACK_R);
    snapCameraToWorldPoint(wp, dist);
  }

  function snapCameraToWorldPoint(wp: THREE.Vector3, dist = FOCUS_CAMERA_DIST) {
    controls.target.copy(wp);
    const dir = wp.clone().normalize();
    camera.position.copy(wp.clone().add(dir.multiplyScalar(dist)));
  }

  function focusOnEvent(ev: GlobeAlertEvent) {
    if (spaceMode && ev.type === "neo") {
      focusSpaceNeo(ev);
      return;
    }
    focusEvent = ev;
    if (spaceMode) {
      eventLockActive = false;
      callbacks.onFocus?.(ev);
      return;
    }
    eventLockActive = true;
    userControlActive = false;
    if (ev.type !== "hurricane") clearStormTrack();
    if (ev.type !== "neo") clearNeoTrack();

    if (ev.type === "neo" && activeNeoTrack) {
      frameEarthAndNeoTrack(activeNeoTrack);
      return;
    }

    if (ev.type === "neo") {
      frameEarthAndNeoTrack(
        activeNeoTrack ?? {
          designation: ev.designation ?? "",
          points: [],
          closest: {
            t: Date.now(),
            lat: ev.lat,
            lon: ev.lon,
            distAu: ev.distAu ?? 0.05,
            distLd: ev.distLd ?? 5,
            deldotKmS: -(ev.vRel ?? 5),
          },
        },
      );
      return;
    }

    if (ev.type === "hurricane" && activeStormTrack) {
      const pos = stormPositionNow(activeStormTrack);
      snapCameraToWorldPoint(latLonToVec3(pos.lat, pos.lon, STORM_TRACK_R), 1.55);
      return;
    }

    const m = markers.get(ev.id);
    if (m) {
      const wp = new THREE.Vector3();
      m.group.getWorldPosition(wp);
      snapCameraToWorldPoint(wp, 1.65);
    }
    callbacks.onFocus?.(ev);
  }

  function returnToNormal() {
    focusEvent = null;
    spaceFocusId = null;
    eventLockActive = false;
    spaceCameraReleased = false;
    clearStormTrack();
    clearNeoTrack();
    controls.target.set(0, 0, 0);
    camera.position.set(0, 0.12, spaceMode ? SPACE_CAMERA_DIST : INITIAL_CAMERA_DIST);
    applyNeoFocusHighlight();
    syncNeoReticles(activeEvents);
  }

  function updateEventLockCamera() {
    if (spaceMode && spaceFocusId && focusEvent?.type === "neo") {
      updateSpaceFocusCamera();
      return;
    }
    if (!eventLockActive || !focusEvent || userControlActive) return;

    if (focusEvent.type === "neo" && activeNeoDisplay && neoHeadMesh) {
      activeNeoDisplay = buildApproachDisplayTrack(activeNeoTrack ?? activeNeoDisplay);
      const pos = interpolateNeoPoint(activeNeoDisplay);
      const neoWp = neoSpacePosition(pos.lat, pos.lon, pos.distLd);
      neoHeadMesh.position.lerp(neoWp, 0.16);
      billboardToCamera(neoHeadMesh, camera);

      if (neoApproachLine) {
        neoApproachLine.geometry.dispose();
        neoApproachLine.geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, 0, 0),
          neoWp.clone(),
        ]);
        neoApproachLine.computeLineDistances();
      }

      if (neoTrailPts) {
        const arr = neoTrailPts.geometry.attributes.position.array as Float32Array;
        const n = arr.length / 3;
        for (let i = n - 1; i > 0; i--) {
          arr[i * 3] = arr[(i - 1) * 3];
          arr[i * 3 + 1] = arr[(i - 1) * 3 + 1];
          arr[i * 3 + 2] = arr[(i - 1) * 3 + 2];
        }
        arr[0] = neoWp.x;
        arr[1] = neoWp.y;
        arr[2] = neoWp.z;
        neoTrailPts.geometry.attributes.position.needsUpdate = true;
      }

      const earthC = new THREE.Vector3(0, 0, 0);
      const frameCenter = neoWp.clone().lerp(earthC, 0.32);
      controls.target.lerp(frameCenter, 0.1);
      const camDist = 1.8 + visualRadiusFromLd(pos.distLd) * 1.35;
      const offset = camera.position.clone().sub(controls.target);
      if (offset.lengthSq() < 0.01) offset.set(0.3, 0.2, 1);
      offset.normalize().multiplyScalar(camDist);
      camera.position.lerp(frameCenter.clone().add(offset), 0.08);
      return;
    }

    if (focusEvent.type === "hurricane" && activeStormTrack && stormHeadMesh) {
      const pos = stormPositionNow(activeStormTrack);
      const wp = latLonToVec3(pos.lat, pos.lon, STORM_TRACK_R);
      controls.target.lerp(wp, 0.12);
      const dir = wp.clone().normalize();
      const desired = wp.clone().add(dir.multiplyScalar(1.55));
      camera.position.lerp(desired, 0.1);
      return;
    }

    const m = markers.get(focusEvent.id);
    if (m) {
      const wp = new THREE.Vector3();
      m.group.getWorldPosition(wp);
      controls.target.lerp(wp, 0.1);
      const dir = wp.clone().normalize();
      camera.position.lerp(wp.clone().add(dir.multiplyScalar(1.65)), 0.1);
    }
  }

  function setNeoOrbitTracks(tracks: Record<string, NeoOrbitTrack>) {
    neoOrbitTracks = tracks;
    for (const [id, entry] of neoReticles) {
      applyOrbitFromTrack(entry, id);
    }
  }

  function setSpaceMode(enabled: boolean) {
    const changed = spaceMode !== enabled;
    spaceMode = enabled;
    moonOrbitG.visible = enabled;
    moonOrbitLine.visible = false;
    weatherOverlay.setVisible(!enabled);
    if (enabled) {
      controls.minDistance = SPACE_MIN_ZOOM_DIST;
      controls.maxDistance = SPACE_MAX_ZOOM_DIST;
      controls.enablePan = true;
      controls.target.set(0, 0, 0);
      camera.position.set(0, 0.12, SPACE_CAMERA_DIST);
      spaceCameraReleased = false;
    } else {
      controls.minDistance = MIN_ZOOM_DIST;
      controls.maxDistance = 22;
      spaceCameraReleased = false;
    }
    if (changed && enabled) clearNeoTrack();
    if (changed) clearNeoReticles();
    syncNeoReticles(activeEvents);
  }

  function onResize() {
    width = container.clientWidth;
    height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  window.addEventListener("resize", onResize);

  function animate() {
    if (disposed) return;
    raf = requestAnimationFrame(animate);
    clock.update();
    const dt = Math.min(clock.getDelta(), 0.1);
    if (!eventLockActive) {
      if (!spaceMode) {
        earthMesh.rotation.y += EARTH_ROT;
        cloudsMesh.rotation.y += EARTH_ROT * 1.35;
      } else {
        earthMesh.rotation.y += EARTH_ROT * 0.15;
        cloudsMesh.rotation.y += EARTH_ROT * 0.2;
      }
      starMesh.rotation.y -= 0.0002;
      if (spaceMode) {
        moonOrbitG.rotation.y += 0.0012;
        moonMesh.rotation.y += 0.0008;
      }
    }
    const cloudMat = cloudsMesh.material as THREE.MeshLambertMaterial;
    cloudMat.opacity = 0.42 + 0.14 * Math.sin(globalTime * 0.45);
    updateEffects(dt);
    weatherOverlay.update(globalTime);

    weatherRefreshTimer += dt;
    if (weatherRefreshTimer > 900) {
      weatherRefreshTimer = 0;
      void refreshWeatherLayer();
    }

    if (activeStormTrack && stormHeadMesh && !eventLockActive) {
      const pos = stormPositionNow(activeStormTrack);
      const target = latLonToVec3(pos.lat, pos.lon, STORM_TRACK_R + 0.003);
      stormHeadMesh.position.lerp(target, 0.12);
      const pulse = 0.65 + 0.35 * Math.sin(globalTime * 4.5);
      (stormHeadMesh.material as THREE.MeshBasicMaterial).opacity = pulse;
    }

    if (shakeIntensity > 0.0005 && !eventLockActive) {
      camera.position.x += (Math.random() - 0.5) * shakeIntensity;
      camera.position.y += (Math.random() - 0.5) * shakeIntensity;
      shakeIntensity *= 0.93;
    }

    updateEventLockCamera();

    for (const mat of neoDashMats) {
      mat.scale = 1 + 0.08 * Math.sin(globalTime * 3.2);
    }
    if (spaceMode) {
      updateNeoReticleLabels();
    } else if (!eventLockActive || focusEvent?.type !== "neo") {
      updateNeoReticleLabels();
    }
    if (!spaceMode) {
      for (const entry of neoReticles.values()) {
        billboardToCamera(entry.group, camera);
        entry.label.sprite.position.set(0, 0.2, 0);
      }
    } else {
      const framePulse = 0.6 + 0.4 * Math.sin(globalTime * 2.4);
      for (const [id, entry] of neoReticles) {
        if (entry.mesh && entry.rotAxis && entry.rotSpeed) {
          entry.mesh.rotateOnAxis(entry.rotAxis, entry.rotSpeed * dt * ASTEROID_SPIN_DISPLAY);
        }
        if (entry.cometTail) {
          const wp = new THREE.Vector3();
          entry.group.getWorldPosition(wp);
          orientCometTail(entry.cometTail, wp);
        }
        if (entry.frame) {
          billboardToCamera(entry.frame, camera);
          const isFocus = spaceFocusId === id;
          if (entry.frameMat) entry.frameMat.opacity = isFocus ? 0 : 0.32 + 0.28 * framePulse;
        }
      }
      applyNeoFocusHighlight();
      updateSelectionFrame();
    }

    if (activeStormTrack && stormHeadMesh && eventLockActive && focusEvent?.type === "hurricane") {
      const pos = stormPositionNow(activeStormTrack);
      const target = latLonToVec3(pos.lat, pos.lon, STORM_TRACK_R + 0.003);
      stormHeadMesh.position.lerp(target, 0.14);
      const pulse = 0.65 + 0.35 * Math.sin(globalTime * 4.5);
      (stormHeadMesh.material as THREE.MeshBasicMaterial).opacity = pulse;
    }

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return {
    syncEvents,
    focusEvent: focusOnEvent,
    returnToNormal,
    showStormTrack,
    clearStormTrack,
    showNeoTrack,
    clearNeoTrack,
    focusNeoEarthFrame,
    focusSpaceNeo,
    clearSpaceNeoFocus,
    flyToLatLon,
    setSpaceMode,
    setNeoOrbitTracks,
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      clearStormTrack();
      clearNeoTrack();
      weatherOverlay.dispose();
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
      activeEffects.forEach((fx) => earthMesh.remove(fx.group));
      markers.forEach((m) => earthMesh.remove(m.group));
      moonGeo.dispose();
      moonMat.dispose();
      moonMap?.dispose();
    },
  };
}
