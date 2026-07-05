import { jplJson } from "./jplApi";

export type ScoutRecord = {
  objectName: string;
  caDistLd: number;
  vInf: number;
  moidAu: number;
  neoScore: number;
  phaScore: number;
  geocentricScore: number;
  h?: number;
  vmag?: number;
  lastRun?: string;
};

type ScoutPayload = {
  count?: string;
  data?: Array<Record<string, string>>;
};

export async function fetchScoutSummary(): Promise<ScoutRecord[]> {
  const payload = await jplJson<ScoutPayload>("/scout.api");
  if (!payload.data?.length) return [];

  const rows: ScoutRecord[] = [];
  for (const row of payload.data) {
    const caDistLd = Number.parseFloat(row.caDist ?? "");
    const vInf = Number.parseFloat(row.vInf ?? "");
    const moidAu = Number.parseFloat(row.moid ?? "");
    const neoScore = Number.parseFloat(row.neoScore ?? "0");
    const phaScore = Number.parseFloat(row.phaScore ?? "0");
    const geocentricScore = Number.parseFloat(row.geocentricScore ?? "0");
    const h = Number.parseFloat(row.H ?? "");
    const vmag = Number.parseFloat(row.Vmag ?? "");

    if (!row.objectName || !Number.isFinite(caDistLd)) continue;

    rows.push({
      objectName: row.objectName,
      caDistLd,
      vInf: Number.isFinite(vInf) ? vInf : 0,
      moidAu: Number.isFinite(moidAu) ? moidAu : 0,
      neoScore: Number.isFinite(neoScore) ? neoScore : 0,
      phaScore: Number.isFinite(phaScore) ? phaScore : 0,
      geocentricScore: Number.isFinite(geocentricScore) ? geocentricScore : 0,
      h: Number.isFinite(h) ? h : undefined,
      vmag: Number.isFinite(vmag) ? vmag : undefined,
      lastRun: row.lastRun,
    });
  }

  return rows
    .filter((r) => r.caDistLd < 15 || r.geocentricScore > 0 || r.neoScore >= 80)
    .sort((a, b) => a.caDistLd - b.caDistLd)
    .slice(0, 8);
}
