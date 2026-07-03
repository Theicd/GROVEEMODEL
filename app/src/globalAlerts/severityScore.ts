import type { GlobeAlertEvent } from "./types";

export type SeverityTier = "low" | "moderate" | "high" | "critical";

export type EventSeverity = {
  score: number;
  tier: SeverityTier;
  label: string;
  bars: number;
};

const TIER_LABELS: Record<SeverityTier, string> = {
  low: "נמוך",
  moderate: "בינוני",
  high: "גבוה",
  critical: "קריטי",
};

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}

function tierFromScore(score: number): SeverityTier {
  if (score >= 78) return "critical";
  if (score >= 55) return "high";
  if (score >= 32) return "moderate";
  return "low";
}

function alertLevelScore(alert?: string): number {
  if (/red/i.test(alert ?? "")) return 88;
  if (/orange/i.test(alert ?? "")) return 58;
  if (/green/i.test(alert ?? "")) return 28;
  return 40;
}

export function getEventSeverity(ev: GlobeAlertEvent): EventSeverity {
  let score = 40;

  if (ev.type === "earthquake" && ev.magnitude != null) {
    score = clamp(((ev.magnitude - 4.0) / 3.0) * 100, 8, 100);
    if (ev.magnitude >= 7.5) score = 100;
    else if (ev.magnitude >= 7) score = Math.max(score, 90);
    else if (ev.magnitude >= 6.5) score = Math.max(score, 82);
  } else if (ev.type === "hurricane" && ev.category != null) {
    score = clamp(ev.category * 20, 20, 100);
  } else if (ev.type === "neo" && ev.distLd != null) {
    if (ev.distLd < 0.5) score = 98;
    else if (ev.distLd < 1) score = 88;
    else if (ev.distLd < 5) score = 68;
    else if (ev.distLd < 20) score = 48;
    else score = 30;
    if (ev.isPha) score = Math.min(100, score + 8);
  } else if (ev.type === "fireball") {
    const kt = ev.impactKt ?? 0;
    score = clamp(40 + kt * 12, 35, 95);
  } else if (ev.type === "tsunami") {
    score = Math.max(alertLevelScore(ev.alertLevel), 72);
  } else if (ev.severity != null) {
    score = clamp(ev.severity * 35, 25, 95);
  } else if (ev.alertLevel) {
    score = alertLevelScore(ev.alertLevel);
  }

  const tier = tierFromScore(score);
  const bars = tier === "critical" ? 4 : tier === "high" ? 3 : tier === "moderate" ? 2 : 1;

  return { score, tier, label: TIER_LABELS[tier], bars };
}
