import { fetchJson } from "../fetchJson";
import { isStaticWebHost } from "../proxyFetch";
import type { SearchSourceResult } from "../types";
import { extractCurrencyPair, type CurrencyPair } from "../queryExtract";

type FrankfurterLatest = {
  amount: number;
  base: string;
  date: string;
  rates: Record<string, number>;
};

type ErApiLatest = {
  result?: string;
  base_code?: string;
  time_last_update_utc?: string;
  rates?: Record<string, number>;
};

const formatFxLines = (
  pair: CurrencyPair,
  rate: number,
  date: string,
  sourceLabel: string,
): string[] => {
  const amount = pair.amount;
  return [
    `תאריך: ${date}`,
    ...(amount != null && amount !== 1
      ? [`${amount} ${pair.from} = ${(rate * amount).toFixed(2)} ${pair.to}`]
      : []),
    `1 ${pair.from} = ${rate} ${pair.to}`,
    `100 ${pair.from} = ${(rate * 100).toFixed(4)} ${pair.to}`,
    `מקור: ${sourceLabel}`,
  ];
};

const fetchFrankfurter = async (pair: CurrencyPair) => {
  const data = await fetchJson<FrankfurterLatest>(
    `https://api.frankfurter.app/latest?from=${pair.from}&to=${pair.to}`,
  );
  const rate = data.rates?.[pair.to];
  if (rate == null) return null;
  return { rate, date: data.date, label: "European Central Bank via Frankfurter" };
};

const fetchErApi = async (pair: CurrencyPair) => {
  const data = await fetchJson<ErApiLatest>(`https://open.er-api.com/v6/latest/${pair.from}`);
  const rate = data.rates?.[pair.to];
  if (rate == null) return null;
  const date =
    data.time_last_update_utc?.replace("UTC", "").trim().slice(0, 10) ??
    new Date().toISOString().slice(0, 10);
  return { rate, date, label: "open.er-api.com (ECB reference)" };
};

export const fetchCurrencySearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "frankfurter-fx" as const;
  const label = "שערי מטבע (Frankfurter)";
  try {
    const pair = extractCurrencyPair(query);
    if (!pair) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא זוהו קודי מטבע (למשל USD ל-ILS)",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const fetchers = isStaticWebHost()
      ? [fetchErApi, fetchFrankfurter]
      : [fetchFrankfurter, fetchErApi];
    let hit: { rate: number; date: string; label: string } | null = null;
    for (const fetcher of fetchers) {
      try {
        hit = await fetcher(pair);
        if (hit) break;
      } catch {
        /* try next source */
      }
    }

    if (!hit) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצא שער",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = formatFxLines(pair, hit.rate, hit.date, hit.label);

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://api.frankfurter.app/latest?from=${pair.from}&to=${pair.to}`,
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
