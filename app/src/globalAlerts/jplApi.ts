import { fetchJson, fetchText } from "../webSearch/fetchJson";

const JPL_BASE = "https://ssd-api.jpl.nasa.gov";

/** JPL asks clients not to fire parallel requests — serialize calls. */
let chain: Promise<unknown> = Promise.resolve();

export function jplSequential<T>(fn: () => Promise<T>): Promise<T> {
  const run = chain.then(fn, fn);
  chain = run.catch(() => undefined);
  return run;
}

export async function jplJson<T>(path: string, timeoutMs = 22_000): Promise<T> {
  return jplSequential(() => fetchJson<T>(`${JPL_BASE}${path}`, undefined, { timeoutMs }));
}

export async function jplText(path: string, timeoutMs = 28_000): Promise<string> {
  return jplSequential(() => fetchText(`${JPL_BASE}${path}`, undefined, { timeoutMs }));
}

/** 1 lunar distance in astronomical units. */
export const AU_PER_LD = 384_400 / 149_597_870.7;

export function auToLd(au: number): number {
  return au / AU_PER_LD;
}

export function parseCadDistAu(raw: string | number): number {
  const n = typeof raw === "number" ? raw : Number.parseFloat(raw);
  return Number.isFinite(n) ? n : NaN;
}

export function parseJplUtcDate(cd: string): number {
  const m = cd.match(
    /(\d{4})-(\w{3})-(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?/,
  );
  if (!m) return Date.parse(cd.replace(" ", "T") + "Z") || Date.now();
  const months: Record<string, number> = {
    Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5,
    Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11,
  };
  const mo = months[m[2]];
  if (mo == null) return Date.now();
  return Date.UTC(
    Number(m[1]),
    mo,
    Number(m[3]),
    Number(m[4]),
    Number(m[5]),
    Number(m[6] ?? 0),
  );
}
