import { getAisStreamApiKey, isAisStreamConfigured } from "../../apiKeys/apiKeyStore";
import { recordProviderUsage } from "../../apiKeys/apiProviderUsage";
import type { ShipBbox } from "../medPorts";
import type { ShipHit } from "../shipAggregate";

export type AisStreamProxyResponse = {
  ok: boolean;
  ships?: Array<{
    name: string;
    lat: number;
    lon: number;
    speed?: number;
    mmsi?: number;
    destination?: string;
    source: "aisstream";
  }>;
  count?: number;
  fetchedAt?: string;
  error?: string;
  warning?: string;
};

const devProxyAvailable = (): boolean =>
  import.meta.env.DEV ||
  (typeof window !== "undefined" &&
    (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"));

/** Fetch live AIS ships for bbox via local dev proxy (POST /api/aisstream/ships). */
export const fetchAisStreamShips = async (
  bbox: ShipBbox,
  options?: { timeoutMs?: number; apiKey?: string },
): Promise<ShipHit[]> => {
  const apiKey = options?.apiKey ?? getAisStreamApiKey();
  if (!apiKey || (!options?.apiKey && !isAisStreamConfigured())) return [];
  if (!devProxyAvailable()) return [];

  try {
    const res = await fetch("/api/aisstream/ships", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        apiKey,
        minLat: bbox.minLat,
        maxLat: bbox.maxLat,
        minLon: bbox.minLon,
        maxLon: bbox.maxLon,
        timeoutMs: options?.timeoutMs ?? 10_000,
      }),
      signal: AbortSignal.timeout((options?.timeoutMs ?? 10_000) + 4000),
    });
    const data = (await res.json()) as AisStreamProxyResponse;
    const rawJson = JSON.stringify(data);
    const hitCount = data.count ?? data.ships?.length ?? 0;
    const ok = res.ok && data.ok && Boolean(data.ships?.length);
    if (!options?.apiKey) {
      recordProviderUsage("aisstream", { ok, hitCount, bytesApprox: rawJson.length });
    }
    if (!res.ok || !data.ok || !data.ships?.length) return [];
    return data.ships.map((s) => ({
      name: s.name,
      lat: s.lat,
      lon: s.lon,
      speed: s.speed,
      destination: s.destination,
      source: "aisstream" as const,
      timestamp: data.fetchedAt ?? new Date().toISOString(),
    }));
  } catch {
    return [];
  }
};

export type AisStreamGlobeResponse = AisStreamProxyResponse & { regions?: number };

/** Multi-region AIS via POST /api/aisstream/globe (Med, Europe, US East, Suez, North Sea). */
export const fetchAisStreamGlobeShips = async (options?: {
  timeoutMs?: number;
  apiKey?: string;
}): Promise<ShipHit[]> => {
  const apiKey = options?.apiKey ?? getAisStreamApiKey();
  if (!apiKey || (!options?.apiKey && !isAisStreamConfigured())) return [];
  if (!devProxyAvailable()) return [];

  try {
    const res = await fetch("/api/aisstream/globe", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ apiKey, timeoutMs: options?.timeoutMs ?? 14_000 }),
      signal: AbortSignal.timeout((options?.timeoutMs ?? 14_000) + 5000),
    });
    const data = (await res.json()) as AisStreamGlobeResponse;
    const rawJson = JSON.stringify(data);
    const hitCount = data.count ?? data.ships?.length ?? 0;
    const ok = res.ok && data.ok && Boolean(data.ships?.length);
    if (!options?.apiKey) {
      recordProviderUsage("aisstream", { ok, hitCount, bytesApprox: rawJson.length });
    }
    if (!res.ok || !data.ok || !data.ships?.length) return [];
    return data.ships.map((s) => ({
      name: s.name,
      lat: s.lat,
      lon: s.lon,
      speed: s.speed,
      destination: s.destination,
      source: "aisstream" as const,
      timestamp: data.fetchedAt ?? new Date().toISOString(),
    }));
  } catch {
    return [];
  }
};

/** Probe AISStream connectivity (for keys panel test button). */
export const probeAisStreamConnection = async (
  apiKey: string,
  bbox: ShipBbox = { minLat: 32.72, maxLat: 32.92, minLon: 34.92, maxLon: 35.12 },
): Promise<{ ok: boolean; count: number; message: string }> => {
  if (!devProxyAvailable()) {
    return { ok: false, count: 0, message: "Proxy זמין רק ב-npm run dev (127.0.0.1:5180)" };
  }
  try {
    const res = await fetch("/api/aisstream/ships", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apiKey, ...bbox, timeoutMs: 12_000 }),
      signal: AbortSignal.timeout(16_000),
    });
    const data = (await res.json()) as AisStreamProxyResponse;
    if (!res.ok || !data.ok) {
      recordProviderUsage("aisstream", { ok: false, bytesApprox: JSON.stringify(data).length });
      return { ok: false, count: 0, message: data.error ?? `HTTP ${res.status}` };
    }
    const n = data.count ?? data.ships?.length ?? 0;
    recordProviderUsage("aisstream", {
      ok: true,
      hitCount: n,
      bytesApprox: JSON.stringify(data).length,
    });
    return {
      ok: n > 0 || !data.warning,
      count: n,
      message:
        n > 0
          ? `✓ ${n} כלי שייט ב-AISStream (מפרץ חיפה)`
          : data.warning
            ? `מחובר — 0 בטווח (${data.warning})`
            : "מחובר — 0 כלי בטווח הבדיקה (נסה שוב או הרחב אזור)",
    };
  } catch (e) {
    return { ok: false, count: 0, message: e instanceof Error ? e.message : "שגיאת רשת" };
  }
};
