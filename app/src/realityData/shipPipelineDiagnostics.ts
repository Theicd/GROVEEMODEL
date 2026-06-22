/**
 * Ship AIS pipeline diagnostics — run via vitest or GET /api/ships/diagnostics (dev).
 * Maps each layer: source fetch → cache caps → map/SERP render limits.
 */

export type ShipDiagLayer = {
  id: string;
  labelHe: string;
  status: "ok" | "warn" | "fail" | "skip";
  detail: string;
  count?: number;
};

export type ShipPipelineReport = {
  at: string;
  layers: ShipDiagLayer[];
  summaryHe: string;
  bottleneck?: string;
};

const DIGITRAFFIC_URL = "https://meri.digitraffic.fi/api/ais/v1/locations";

export const DIAG_LIMITS = {
  digitrafficStoreCap: 8000,
  aisStreamGlobeCap: 1500,
  snapshotCacheCap: 500,
  serpCardCap: 64,
  mapRenderFar: 1200,
  mapRenderMid: 900,
  mapRenderNear: 2200,
} as const;

export async function probeDigitrafficLocations(timeoutMs = 35_000): Promise<{
  ok: boolean;
  count: number;
  error?: string;
  latRange?: [number, number];
}> {
  try {
    const res = await fetch(DIGITRAFFIC_URL, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { ok: false, count: 0, error: `HTTP ${res.status}` };
    const geo = (await res.json()) as { features?: Array<{ geometry?: { coordinates?: [number, number] } }> };
    const features = geo.features ?? [];
    const lats = features.map((f) => f.geometry?.coordinates?.[1]).filter((n): n is number => n != null);
    return {
      ok: features.length > 100,
      count: features.length,
      latRange: lats.length ? [Math.min(...lats), Math.max(...lats)] : undefined,
    };
  } catch (e) {
    return { ok: false, count: 0, error: e instanceof Error ? e.message : "fetch failed" };
  }
};

export async function probeAisStreamGlobeViaProxy(
  apiKey: string,
  origin = "http://127.0.0.1:5180",
  timeoutMs = 24_000,
): Promise<{ ok: boolean; count: number; error?: string; warning?: string }> {
  if (!apiKey.trim()) return { ok: false, count: 0, error: "no API key" };
  try {
    const res = await fetch(`${origin}/api/aisstream/globe`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ apiKey, timeoutMs }),
      signal: AbortSignal.timeout(timeoutMs + 8000),
    });
    const data = (await res.json()) as {
      ok?: boolean;
      count?: number;
      ships?: unknown[];
      error?: string;
      warning?: string;
    };
    if (!res.ok || !data.ok) return { ok: false, count: 0, error: data.error ?? `HTTP ${res.status}` };
    const count = data.count ?? data.ships?.length ?? 0;
    return { ok: count > 0, count, warning: data.warning };
  } catch (e) {
    return { ok: false, count: 0, error: e instanceof Error ? e.message : "proxy unreachable — npm run dev?" };
  }
};

export function buildPipelineReport(input: {
  digitraffic: Awaited<ReturnType<typeof probeDigitrafficLocations>>;
  aisStream?: { ok: boolean; count: number; error?: string; warning?: string };
  liveStoreCount?: number;
  renderCount?: number;
  hostFlags?: { port?: string; staticHost?: boolean; localDev?: boolean; proxyUsed?: boolean };
}): ShipPipelineReport {
  const layers: ShipDiagLayer[] = [];

  layers.push({
    id: "digitraffic-api",
    labelHe: "Digitraffic API (Baltic/North Europe)",
    status: input.digitraffic.ok ? "ok" : "fail",
    count: input.digitraffic.count,
    detail: input.digitraffic.ok
      ? `✓ ${input.digitraffic.count} כלי · lat ${input.digitraffic.latRange?.map((n) => n.toFixed(1)).join("–") ?? "?"}`
      : `✗ ${input.digitraffic.error ?? "no data"} — לא גלובלי!`,
  });

  if (input.aisStream) {
    layers.push({
      id: "aisstream-globe",
      labelHe: "AISStream globe (proxy)",
      status: input.aisStream.ok ? "ok" : input.aisStream.error?.includes("npm run dev") ? "fail" : "warn",
      count: input.aisStream.count,
      detail: input.aisStream.ok
        ? `✓ ${input.aisStream.count} כלי · מקס ${DIAG_LIMITS.aisStreamGlobeCap}`
        : `✗ ${input.aisStream.error ?? "0 ships"}${input.aisStream.warning ? ` (${input.aisStream.warning})` : ""}`,
    });
  }

  if (input.hostFlags) {
    const h = input.hostFlags;
    const badStatic = h.staticHost && h.localDev;
    layers.push({
      id: "host-routing",
      labelHe: "ניתוב localhost / proxy",
      status: badStatic ? "fail" : h.localDev ? "ok" : "warn",
      detail: badStatic
        ? `✗ פורט ${h.port} מסומן staticHost — Digitraffic לא עובר /api/proxy`
        : h.localDev
          ? `✓ localDev port=${h.port ?? "?"} proxy=${h.proxyUsed ? "yes" : "vite"}`
          : `אירוח static — relay CORS בלבד`,
    });
  }

  layers.push({
    id: "ui-caps",
    labelHe: "תקרות ממשק (by design)",
    status: "warn",
    detail: `מפה עד ${DIAG_LIMITS.mapRenderNear} · SERP ${DIAG_LIMITS.serpCardCap} כרטיסים · מטמון ${DIAG_LIMITS.snapshotCacheCap}`,
  });

  if (input.liveStoreCount != null) {
    layers.push({
      id: "live-store",
      labelHe: "live.ships במפה",
      status: input.liveStoreCount > 500 ? "ok" : input.liveStoreCount > 50 ? "warn" : "fail",
      count: input.liveStoreCount,
      detail:
        input.liveStoreCount > 500
          ? `✓ ${input.liveStoreCount} במטמון`
          : `✗ רק ${input.liveStoreCount} — fetch נכשל או proxy`,
    });
  }

  const fail = layers.find((l) => l.status === "fail");
  const bottleneck =
    fail?.id ??
    (input.digitraffic.ok && (input.aisStream?.count ?? 0) < 20 ? "aisstream-globe" : undefined);

  let summaryHe = "בדיקת צינור AIS";
  if (fail?.id === "host-routing") {
    summaryHe = "בottleneck: localhost לא מזוהה כ-dev — Digitraffic לא נטען (5180)";
  } else if (fail?.id === "digitraffic-api") {
    summaryHe = "Digitraffic לא נגיש — CORS/proxy/timeout";
  } else if (bottleneck === "aisstream-globe") {
    summaryHe = "AISStream מחזיר מעט — בדוק מפתח / המתן 22s";
  } else if (input.liveStoreCount != null && input.liveStoreCount < 100) {
    summaryHe = `רק ${input.liveStoreCount} ספינות במפה — לא אלפים`;
  } else {
    summaryHe = `Digitraffic ~${input.digitraffic.count} (Baltic) + AISStream ~${input.aisStream?.count ?? "?"}`;
  }

  return { at: new Date().toISOString(), layers, summaryHe, bottleneck };
};

export async function runShipPipelineDiagnostics(options?: {
  aisStreamKey?: string;
  devOrigin?: string;
}): Promise<ShipPipelineReport> {
  const digitraffic = await probeDigitrafficLocations();
  let aisStream: Awaited<ReturnType<typeof probeAisStreamGlobeViaProxy>> | undefined;
  if (options?.aisStreamKey) {
    aisStream = await probeAisStreamGlobeViaProxy(
      options.aisStreamKey,
      options.devOrigin ?? "http://127.0.0.1:5180",
    );
  }
  return buildPipelineReport({ digitraffic, aisStream });
}
