import { fetchJson, fetchText } from "../webSearch/fetchJson";

export type IntelTickerItem = {
  id: string;
  severity: number;
  tag: string;
  text: string;
  time: string;
  ts?: number;
  icon?: string;
  category?: string;
  lat?: number;
  lon?: number;
  geo?: { lat: number; lon: number };
};

export type IntelHeadline = {
  id: string;
  text: string;
  severity: number;
};

export type IntelFlashAlert = {
  id: string;
  severity: number;
  title: string;
  body: string;
  category: string;
  lat?: number;
  lon?: number;
  place?: string;
  magnitude?: number;
  depth?: number;
  source?: string;
  recommendedAction?: string;
  eventTime?: string;
};

export type GlobeIntelSnapshot = {
  tickers: IntelTickerItem[];
  headlines: IntelHeadline[];
  flash: IntelFlashAlert | null;
};

const colorSeverity = (mag: number): number => {
  if (mag >= 6.5) return 5;
  if (mag >= 5) return 4;
  if (mag >= 4) return 3;
  return 2;
};

export async function fetchGlobeIntelSnapshot(): Promise<GlobeIntelSnapshot> {
  const tickers: IntelTickerItem[] = [];
  const now = new Date().toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });

  const [eqRes, alertRes, spaceRes] = await Promise.allSettled([
    fetchJson<{ features?: { id: string; properties: { mag?: number; place?: string; time?: number } }[] }>(
      "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
    ),
    fetchText("https://api.tzevaadom.co.il/notifications"),
    fetchJson<{ kp_index?: number; message?: string }>(
      "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    ),
  ]);

  if (eqRes.status === "fulfilled") {
    for (const f of eqRes.value.features ?? []) {
      const mag = f.properties.mag ?? 0;
      const sev = colorSeverity(mag);
      const ts = f.properties.time ?? Date.now();
      const text = `רעידת אדמה M${mag.toFixed(1)} · ${f.properties.place ?? "לא ידוע"}`;
      tickers.push({
        id: `eq-${f.id}`,
        severity: sev,
        tag: mag >= 5 ? "BREAKING" : "רעידה",
        text,
        time: new Date(ts).toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" }),
        ts,
        icon: "🌍",
        category: "SEISMIC",
      });
    }
  }

  if (alertRes.status === "fulfilled") {
    try {
      const parsed = JSON.parse(alertRes.value) as { alerts?: unknown[] } | unknown[];
      const alerts = Array.isArray(parsed) ? parsed : (parsed.alerts ?? []);
      if (alerts.length) {
        const areas = alerts.slice(0, 5).map((a) => {
          const o = a as Record<string, unknown>;
          return String(o.title ?? o.area ?? o.city ?? o.data ?? "אזור");
        });
        const text = `צבע אדום · ${areas.join(", ")}`;
        tickers.unshift({
          id: `il-${Date.now()}`,
          severity: 5,
          tag: "צבע אדום",
          text,
          time: now,
          ts: Date.now(),
          icon: "🚨",
          category: "ISRAEL",
        });
      } else {
        tickers.push({
          id: "il-clear",
          severity: 1,
          tag: "ישראל",
          text: "✅ אין התרעות צבע אדום פעילות",
          time: now,
        });
      }
    } catch {
      /* ignore parse */
    }
  }

  if (spaceRes.status === "fulfilled") {
    const rows = spaceRes.value as unknown;
    let kp = 0;
    if (Array.isArray(rows) && rows.length > 1) {
      const last = rows[rows.length - 1] as string[];
      kp = Number(last[1]) || 0;
    }
    if (kp >= 4) {
      const text = `מזג חלל · Kp=${kp}${kp >= 5 ? " · סיכון גיאומגנטי" : ""}`;
      tickers.push({
        id: `kp-${kp}`,
        severity: kp >= 6 ? 4 : 3,
        tag: "חלל",
        text,
        time: now,
      });
    }
  }

  if (!tickers.length) {
    tickers.push({
      id: "idle",
      severity: 1,
      tag: "LIVE",
      text: "מנטר עולם · ממתין לנתונים חיים…",
      time: now,
    });
  }

  const headlines: IntelHeadline[] = [
    { id: "hl-live", text: "REALITY LIVE · מעקב עולמי בזמן אמת", severity: 2 },
  ];

  return { tickers, headlines, flash: null };
}
