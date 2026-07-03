import { jplJson } from "./jplApi";

export type FireballRecord = {
  id: string;
  time: number;
  lat: number;
  lon: number;
  altKm: number;
  energyJ1e10: number;
  impactKt: number;
  vx: number;
  vy: number;
  vz: number;
};

type FireballPayload = {
  count?: string;
  fields?: string[];
  data?: Array<Array<string | number | null>>;
};

function parseFireballDate(raw: string): number {
  const t = Date.parse(raw.replace(" ", "T") + "Z");
  return Number.isFinite(t) ? t : Date.now();
}

function parseLatLon(lat: string, lon: string, latDir?: string | null, lonDir?: string | null): { lat: number; lon: number } | null {
  let la = Number.parseFloat(lat);
  let lo = Number.parseFloat(lon);
  if (!Number.isFinite(la) || !Number.isFinite(lo)) return null;
  if (latDir === "S") la = -la;
  if (lonDir === "W") lo = -lo;
  return { lat: la, lon: lo };
}

export async function fetchRecentFireballs(opts?: { limit?: number; daysBack?: number }): Promise<FireballRecord[]> {
  const limit = opts?.limit ?? 25;
  const daysBack = opts?.daysBack ?? 90;
  const start = new Date();
  start.setUTCDate(start.getUTCDate() - daysBack);
  const dateMin = start.toISOString().slice(0, 10);

  const q = new URLSearchParams({
    "req-loc": "true",
    limit: String(limit),
    "date-min": dateMin,
    sort: "-date",
  });

  const payload = await jplJson<FireballPayload>(`/fireball.api?${q}`);
  const fields = payload.fields ?? [];
  if (!fields.length || !payload.data?.length) return [];

  const idx = (name: string) => fields.indexOf(name);
  const iDate = idx("date");
  const iLat = idx("lat");
  const iLon = idx("lon");
  const iLatDir = idx("lat-dir");
  const iLonDir = idx("lon-dir");
  const iAlt = idx("alt");
  const iEnergy = idx("energy");
  const iImpact = idx("impact-e");
  const iVx = idx("vx");
  const iVy = idx("vy");
  const iVz = idx("vz");

  const out: FireballRecord[] = [];
  for (const [i, row] of payload.data.entries()) {
    const pos = parseLatLon(
      String(row[iLat] ?? ""),
      String(row[iLon] ?? ""),
      row[iLatDir] as string | null,
      row[iLonDir] as string | null,
    );
    if (!pos) continue;
    const date = String(row[iDate] ?? "");
    const impactKt = Number.parseFloat(String(row[iImpact] ?? "0"));
    const energy = Number.parseFloat(String(row[iEnergy] ?? "0"));
    const vx = Number.parseFloat(String(row[iVx] ?? "0"));
    const vy = Number.parseFloat(String(row[iVy] ?? "0"));
    const vz = Number.parseFloat(String(row[iVz] ?? "0"));
    const altKm = Number.parseFloat(String(row[iAlt] ?? "0"));

    out.push({
      id: `fireball-${date}-${i}`,
      time: parseFireballDate(date),
      lat: pos.lat,
      lon: pos.lon,
      altKm: Number.isFinite(altKm) ? altKm : 0,
      energyJ1e10: Number.isFinite(energy) ? energy : 0,
      impactKt: Number.isFinite(impactKt) ? impactKt : 0,
      vx: Number.isFinite(vx) ? vx : 0,
      vy: Number.isFinite(vy) ? vy : 0,
      vz: Number.isFinite(vz) ? vz : 0,
    });
  }
  return out;
}
