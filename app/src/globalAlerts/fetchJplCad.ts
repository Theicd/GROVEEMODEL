import { auToLd, jplJson, parseCadDistAu, parseJplUtcDate } from "./jplApi";

export type CadRecord = {
  des: string;
  fullname?: string;
  approachTime: number;
  approachLabel: string;
  distAu: number;
  distMinAu: number;
  distMaxAu: number;
  distLd: number;
  vRel: number;
  vInf: number;
  h?: number;
  diameterKm?: number;
};

type CadPayload = {
  count?: string;
  fields?: string[];
  data?: Array<Array<string | number>>;
};

function fieldIndex(fields: string[], name: string): number {
  return fields.indexOf(name);
}

export async function fetchNeoCloseApproaches(opts?: {
  daysAhead?: number;
  distMaxAu?: number;
  limit?: number;
}): Promise<CadRecord[]> {
  const days = opts?.daysAhead ?? 14;
  const distMax = opts?.distMaxAu ?? 0.15;
  const limit = opts?.limit ?? 18;

  const end = new Date();
  end.setUTCDate(end.getUTCDate() + days);
  const dateMax = end.toISOString().slice(0, 10);

  const q = new URLSearchParams({
    "date-min": "now",
    "date-max": dateMax,
    "dist-max": String(distMax),
    neo: "true",
    sort: "dist",
    limit: String(limit),
    diameter: "true",
    fullname: "true",
  });

  const payload = await jplJson<CadPayload>(`/cad.api?${q}`);
  const fields = payload.fields ?? [];
  if (!fields.length || !payload.data?.length) return [];

  const iDes = fieldIndex(fields, "des");
  const iCd = fieldIndex(fields, "cd");
  const iDist = fieldIndex(fields, "dist");
  const iDistMin = fieldIndex(fields, "dist_min");
  const iDistMax = fieldIndex(fields, "dist_max");
  const iVrel = fieldIndex(fields, "v_rel");
  const iVinf = fieldIndex(fields, "v_inf");
  const iH = fieldIndex(fields, "h");
  const iDiam = fieldIndex(fields, "diameter");
  const iName = fieldIndex(fields, "fullname");

  const out: CadRecord[] = [];
  for (const row of payload.data) {
    const des = String(row[iDes] ?? "");
    const cd = String(row[iCd] ?? "");
    const distAu = parseCadDistAu(row[iDist] as string);
    if (!des || !Number.isFinite(distAu)) continue;

    const distMinAu = parseCadDistAu(row[iDistMin] as string);
    const distMaxAu = parseCadDistAu(row[iDistMax] as string);
    const vRel = Number.parseFloat(String(row[iVrel] ?? ""));
    const vInf = Number.parseFloat(String(row[iVinf] ?? ""));
    const hRaw = row[iH];
    const h = hRaw != null ? Number.parseFloat(String(hRaw)) : undefined;
    const dRaw = row[iDiam];
    const diameterKm = dRaw != null && String(dRaw) !== "" ? Number.parseFloat(String(dRaw)) : undefined;
    const fullname = iName >= 0 ? String(row[iName] ?? "") : undefined;

    out.push({
      des,
      fullname: fullname || undefined,
      approachTime: parseJplUtcDate(cd),
      approachLabel: cd,
      distAu,
      distMinAu: Number.isFinite(distMinAu) ? distMinAu : distAu,
      distMaxAu: Number.isFinite(distMaxAu) ? distMaxAu : distAu,
      distLd: auToLd(distAu),
      vRel: Number.isFinite(vRel) ? vRel : 0,
      vInf: Number.isFinite(vInf) ? vInf : 0,
      h: Number.isFinite(h!) ? h : undefined,
      diameterKm: Number.isFinite(diameterKm!) ? diameterKm : undefined,
    });
  }
  return out;
}
