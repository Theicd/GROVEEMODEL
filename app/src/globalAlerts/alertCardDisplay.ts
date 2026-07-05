import { parseWindKmh } from "./hurricaneIntensity";
import { inferSpectralType, inferAsteroidShape, SPECTRAL_TYPES, SHAPE_LABELS } from "./spaceObjectVisuals";
import { EVENT_TYPE_LABELS, type GlobeAlertEvent } from "./types";

export type AlertCardDisplay = {
  headline: string;
  chips: string[];
  region?: string;
  regionLtr?: boolean;
  detail?: string;
};

function splitNameRegion(location: string): { name: string; region?: string } {
  const parts = location.split(" · ").map((p) => p.trim()).filter(Boolean);
  if (parts.length <= 1) return { name: parts[0] ?? location };
  return { name: parts[0], region: parts.slice(1).join(" · ") };
}

function stormKindLabel(severityText?: string): string {
  if (!severityText) return "סופה";
  if (/hurricane|typhoon/i.test(severityText)) return "הוריקן";
  if (/tropical storm/i.test(severityText)) return "סערה טропית";
  if (/tropical cyclone/i.test(severityText)) return "צиклון טропי";
  if (/cyclone/i.test(severityText)) return "ציקלון";
  return "סופה";
}

function alertChip(label?: string): string | null {
  const t = label?.trim();
  if (!t) return null;
  if (/^green$/i.test(t)) return "ירוק";
  if (/^orange$/i.test(t)) return "כתום";
  if (/^red$/i.test(t)) return "אדום";
  return t;
}

/** Card copy — headline first (RTL), then chips, then labeled region. */
export function formatAlertCardDisplay(ev: GlobeAlertEvent): AlertCardDisplay {
  const typeLabel = EVENT_TYPE_LABELS[ev.type]?.label ?? "התרעה";

  if (ev.type === "hurricane") {
    const name = ev.location.trim();
    const region = ev.regionLabel?.trim();
    const cat = ev.category != null ? `קטגוריה ${ev.category}` : null;
    const kind = stormKindLabel(ev.severityText);
    const wind = parseWindKmh(ev.severityText);
    const chips = [kind, cat, alertChip(ev.alertLevel)].filter(Boolean) as string[];
    return {
      headline: name,
      chips,
      region,
      regionLtr: true,
      detail: wind != null ? `רוח מקסימלית ~${Math.round(wind)} קמ"ש` : undefined,
    };
  }

  if (ev.type === "earthquake" || ev.type === "tsunami") {
    const mag = ev.magnitude != null ? `M${ev.magnitude.toFixed(1)}` : "";
    const place = ev.location.trim();
    const chips = [typeLabel].filter(Boolean);
    return {
      headline: mag || typeLabel,
      chips: mag ? chips : [],
      region: place,
      regionLtr: true,
      detail: ev.depth != null ? `עומק ${ev.depth.toFixed(0)} ק"מ` : undefined,
    };
  }

  if (ev.type === "neo" || ev.type === "fireball") {
    const name = ev.designation ?? ev.location.trim();
    const spectral = SPECTRAL_TYPES[inferSpectralType(ev)].name.split(" (")[0];
    const shape = SHAPE_LABELS[inferAsteroidShape(ev)];
    const chips = [
      spectral,
      shape,
      ev.showcaseNeo ? "מחזורי" : typeLabel,
      ev.distLd != null ? `${ev.distLd.toFixed(1)} LD` : null,
      ev.isPha ? "PHA" : null,
    ].filter(Boolean) as string[];
    return {
      headline: name,
      chips,
      region: ev.location !== name ? ev.location : undefined,
      regionLtr: true,
      detail: ev.showcaseNeo ? ev.severityText : undefined,
    };
  }

  if (ev.source === "gdacs") {
    const { name, region } = splitNameRegion(ev.location);
    const chips = [typeLabel, alertChip(ev.alertLevel)].filter(Boolean) as string[];
    return {
      headline: name,
      chips,
      region: ev.regionLabel?.trim() || region,
      regionLtr: true,
      detail: ev.severityText?.trim(),
    };
  }

  return {
    headline: ev.location.trim() || typeLabel,
    chips: [typeLabel],
    regionLtr: /[a-z]/i.test(ev.location),
  };
}

/** One-line summary for collapsed peek. */
export function formatAlertCardPeek(ev: GlobeAlertEvent): string {
  const d = formatAlertCardDisplay(ev);
  const parts = [d.headline, ...d.chips.slice(0, 2)];
  if (d.region) parts.push(d.region);
  return parts.filter(Boolean).join(" · ");
}
