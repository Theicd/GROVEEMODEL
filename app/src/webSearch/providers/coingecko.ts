import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

const COIN_IDS: Record<string, string> = {
  bitcoin: "bitcoin",
  ביטקוין: "bitcoin",
  btc: "bitcoin",
  ethereum: "ethereum",
  eth: "ethereum",
  איתריום: "ethereum",
};

export const fetchCoinGeckoSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "coingecko" as const;
  const label = "CoinGecko (קריפטו)";
  try {
    const lower = query.toLowerCase();
    let coinId = "bitcoin";
    for (const [key, id] of Object.entries(COIN_IDS)) {
      if (lower.includes(key)) {
        coinId = id;
        break;
      }
    }

    const data = await fetchJson<Record<string, { usd?: number; ils?: number; usd_24h_change?: number }>>(
      `https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd,ils&include_24hr_change=true`,
    );
    const row = data[coinId];
    if (!row?.usd) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצא מחיר",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      `${coinId}: $${row.usd} USD`,
      row.ils != null ? `≈ ₪${row.ils.toLocaleString("he-IL")}` : "",
      row.usd_24h_change != null ? `שינוי 24h: ${row.usd_24h_change.toFixed(2)}%` : "",
    ].filter(Boolean);

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://www.coingecko.com/en/coins/${coinId}`,
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
