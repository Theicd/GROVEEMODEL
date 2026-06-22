export type GlobeLayer =
  | "aviation"
  | "earthquake"
  | "weather"
  | "marine"
  | "marine_infra"
  | "iss"
  | "satellite"
  | "ships"
  | "israel_alerts";

export type GlobeMapLayer = Exclude<GlobeLayer, "iss">;

export type GlobeLayersState = Record<GlobeMapLayer, boolean>;

export type GlobeRoutePoint = { lat: number; lon: number; label?: string };

export type GlobeCommand =
  | { type: "flyTo"; lat: number; lon: number; alt?: number; label?: string; presentation?: boolean }
  | { type: "flyToAlert"; lat: number; lon: number; severity?: number; category?: string; alt?: number }
  | { type: "focusPlace"; name: string; alt?: number }
  | { type: "focusPlaceQuiet"; name: string; alt?: number; presentation?: boolean }
  | { type: "drawRoute"; points: GlobeRoutePoint[]; label?: string; presentation?: boolean }
  | { type: "setPresentationMode"; on: boolean }
  | { type: "setQuietAlerts"; on: boolean }
  | { type: "showLayer"; layer: GlobeLayer }
  | { type: "focusEarthquakes" }
  | { type: "focusIsrael" }
  | { type: "globe3d" }
  | { type: "initSound" }
  | { type: "playSound"; kind: "critical" | "info" };

function normalizeBase(base: string | undefined): string {
  const b = typeof base === "string" && base.length > 0 ? base : "./";
  return b.endsWith("/") ? b : `${b}/`;
}

export function resolveRealityEmbedSrc(): string {
  const base = normalizeBase(import.meta.env.BASE);
  const rel = `${base}reality/israel.html?embed=grovee`;
  if (typeof window === "undefined") return rel;
  try {
    return new URL(rel, window.location.href).href;
  } catch {
    return rel;
  }
}

/** Lazy — avoids module-load crash when env/window not ready. */
export function getRealityEmbedSrc(): string {
  return resolveRealityEmbedSrc();
}

export function sendGlobeCommand(iframe: HTMLIFrameElement | null, command: GlobeCommand): void {
  if (!iframe?.contentWindow) return;
  iframe.contentWindow.postMessage(
    { source: "grovee", type: command.type, payload: command },
    "*",
  );
}

export function sendGlobeLayerToggle(iframe: HTMLIFrameElement | null, layer: GlobeMapLayer): void {
  if (!iframe?.contentWindow) return;
  iframe.contentWindow.postMessage(
    { source: "grovee", type: "toggleLayer", payload: { layer } },
    "*",
  );
}

export type UserRegionPayload = {
  countryCode: string;
  name?: string;
  lat: number;
  lon: number;
};

export function sendGlobeUserRegion(
  iframe: HTMLIFrameElement | null,
  region: UserRegionPayload,
): void {
  if (!iframe?.contentWindow) return;
  iframe.contentWindow.postMessage(
    { source: "grovee", type: "setUserRegion", payload: region },
    "*",
  );
}
