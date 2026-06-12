import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

type CoinSpec = { id: string; label: string };

const COIN_HINTS: Array<{ re: RegExp; coin: CoinSpec }> = [
  { re: /bitcoin|ביטקוין|btc/i, coin: { id: "bitcoin", label: "Bitcoin" } },
  { re: /\bethereum\b|את(?:ריום)?|eth\b/i, coin: { id: "ethereum", label: "Ethereum" } },
  { re: /solana|\bsol\b/i, coin: { id: "solana", label: "Solana" } },
  { re: /dogecoin|\bdoge\b/i, coin: { id: "dogecoin", label: "Dogecoin" } },
  { re: /(?:מחיר|price).*(?:זהב|gold)|\bgold\b|xau/i, coin: { id: "pax-gold", label: "Gold (PAXG proxy)" } },
  { re: /crypto|קריפטו|מטבע(?:ות)?\s*דיגיט/i, coin: { id: "bitcoin", label: "Bitcoin" } },
];

const extractCoins = (query: string): CoinSpec[] => {
  const found = new Map<string, CoinSpec>();
  for (const { re, coin } of COIN_HINTS) {
    if (re.test(query)) found.set(coin.id, coin);
  }
  return [...found.values()];
};

export const fetchCoinGeckoSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "coingecko" as const;
  const label = "CoinGecko";
  const coins = extractCoins(query);
  if (!coins.length) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "לא זוהה נכס קריפטו/זהב — נסה SearXNG למניות",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  try {
    const ids = coins.map((c) => c.id).join(",");
    const url = `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd,ils&include_24hr_change=true`;
    const data = await fetchJson<Record<string, { usd?: number; ils?: number; usd_24h_change?: number }>>(url);
    const lines: string[] = [];
    for (const coin of coins) {
      const row = data[coin.id];
      if (!row?.usd) continue;
      const ch = row.usd_24h_change != null ? ` (${row.usd_24h_change >= 0 ? "+" : ""}${row.usd_24h_change.toFixed(2)}% 24h)` : "";
      lines.push(
        `- ${coin.label}: $${row.usd.toLocaleString("en-US")}${row.ils ? ` · ₪${row.ils.toLocaleString("he-IL")}` : ""}${ch}`,
      );
    }
    if (!lines.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "CoinGecko לא החזיר מחיר",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://www.coingecko.com/en/coins/${coins[0].id}`,
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
