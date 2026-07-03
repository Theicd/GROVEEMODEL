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
