/** Live countdown until close approach — updates every second in the UI. */
export function formatNeoCountdown(approachTime: number, now = Date.now()): string {
  const ms = approachTime - now;
  if (ms <= 0) return "עבר את נקודת הקרבה";
  const maxMs = 14 * 24 * 3_600_000;
  if (ms > maxMs) return ">14 ימים";
  const totalSec = Math.floor(ms / 1000);
  const d = Math.floor(totalSec / 86_400);
  const h = Math.floor((totalSec % 86_400) / 3_600);
  const m = Math.floor((totalSec % 3_600) / 60);
  const s = totalSec % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  if (d > 0) return `${d}י ${pad(h)}:${pad(m)}:${pad(s)}`;
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

/** Human-readable time until NEO close approach. */
export function formatNeoEta(approachTime: number): string {
  const ms = approachTime - Date.now();
  if (ms <= 0) return "עבר";
  const h = ms / 3_600_000;
  if (h < 1) return `~${Math.max(1, Math.round(ms / 60_000))} דק'`;
  if (h < 48) return `~${Math.round(h)} שע'`;
  const d = Math.round(h / 24);
  return `~${d} ימים`;
}

export function neoSeverityLine(ev: {
  distLd?: number;
  vRel?: number;
  approachTime?: number;
  diameterKm?: number;
}): string {
  const parts: string[] = [];
  if (ev.distLd != null) parts.push(`${ev.distLd.toFixed(2)} LD`);
  if (ev.vRel != null) parts.push(`${ev.vRel.toFixed(1)} km/s`);
  if (ev.diameterKm != null) parts.push(`Ø ${ev.diameterKm.toFixed(2)} km`);
  const eta = ev.approachTime != null ? formatNeoEta(ev.approachTime) : "";
  if (eta) parts.push(eta);
  return parts.join(" · ");
}
