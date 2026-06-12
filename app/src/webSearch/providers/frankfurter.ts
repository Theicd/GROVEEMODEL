import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractCurrencyPair } from "../queryExtract";

type FrankfurterLatest = {
  amount: number;
  base: string;
  date: string;
  rates: Record<string, number>;
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

    const data = await fetchJson<FrankfurterLatest>(
      `https://api.frankfurter.app/latest?from=${pair.from}&to=${pair.to}`,
    );
    const rate = data.rates?.[pair.to];
    if (rate == null) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצא שער",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      `תאריך: ${data.date}`,
      `1 ${pair.from} = ${rate} ${pair.to}`,
      `100 ${pair.from} = ${(rate * 100).toFixed(4)} ${pair.to}`,
      `מקור: European Central Bank via Frankfurter`,
    ];

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
