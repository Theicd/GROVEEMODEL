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

type BinanceTicker = { symbol?: string; price?: string };

const fetchBinanceUsd = async (coinId: string): Promise<number | null> => {
  const symbol = coinId === "ethereum" ? "ETHUSDT" : "BTCUSDT";
  const data = await fetchJson<BinanceTicker>(
    `https://api.binance.com/api/v3/ticker/price?symbol=${symbol}`,
  );
  const price = data.price != null ? parseFloat(data.price) : NaN;
  return Number.isFinite(price) ? price : null;
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

    let usd: number | null = null;
    let ils: number | null = null;
    let change24: number | undefined;
    let sourceNote = "CoinGecko";

    try {
      const data = await fetchJson<Record<string, { usd?: number; ils?: number; usd_24h_change?: number }>>(
        `https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd,ils&include_24hr_change=true`,
      );
      const row = data[coinId];
      if (row?.usd) {
        usd = row.usd;
        ils = row.ils ?? null;
        change24 = row.usd_24h_change;
      }
    } catch {
      /* fallback below */
    }

    if (usd == null) {
      usd = await fetchBinanceUsd(coinId);
      sourceNote = "Binance (BTCUSDT/ETHUSDT)";
      if (usd != null) {
        try {
          const fx = await fetchJson<{ rates?: { ILS?: number } }>(
            "https://open.er-api.com/v6/latest/USD",
          );
          const ilsRate = fx.rates?.ILS;
          if (ilsRate != null) ils = Math.round(usd * ilsRate);
        } catch {
          /* USD only */
        }
      }
    }

    if (usd == null) {
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
      `${coinId}: $${usd} USD`,
      ils != null ? `≈ ₪${ils.toLocaleString("he-IL")}` : "",
      change24 != null ? `שינוי 24h: ${change24.toFixed(2)}%` : "",
      `מקור: ${sourceNote}`,
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
