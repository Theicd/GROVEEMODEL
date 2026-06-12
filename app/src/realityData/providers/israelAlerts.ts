import { fetchText } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

const TZEVA_URL = "https://api.tzevaadom.co.il/notifications";

export const fetchIsraelAlertsSearch = async (_query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "israel-alerts" as const;
  const label = "התרעות ישראל (צבע אדום)";
  try {
    const raw = await fetchText(TZEVA_URL);
    let alerts: unknown[] = [];
    try {
      const parsed = JSON.parse(raw) as { alerts?: unknown[] } | unknown[];
      alerts = Array.isArray(parsed) ? parsed : (parsed.alerts ?? []);
    } catch {
      return {
        provider,
        label,
        ok: true,
        text: "אין התרעות פעילות כרגע (מקור: Tzeva Adom)",
        url: TZEVA_URL,
        latencyMs: Math.round(performance.now() - started),
      };
    }
    if (!alerts.length) {
      return {
        provider,
        label,
        ok: true,
        text: "✅ אין התרעות פעילות כרגע בישראל",
        url: TZEVA_URL,
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const lines = [
      `התרעות פעילות: ${alerts.length}`,
      ...alerts.slice(0, 8).map((a, i) => {
        const o = a as Record<string, unknown>;
        const area = o.title ?? o.area ?? o.city ?? o.data ?? JSON.stringify(a).slice(0, 80);
        return `${i + 1}. ${String(area)}`;
      }),
    ];
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: TZEVA_URL,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
