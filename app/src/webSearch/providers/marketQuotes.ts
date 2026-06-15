import { fetchJson, fetchText } from "../fetchJson";
import { isStaticWebHost, proxyAwareFetch } from "../proxyFetch";
import type { SearchSourceResult } from "../types";

type YahooChart = {
  chart?: {
    result?: Array<{
      meta?: {
        symbol?: string;
        shortName?: string;
        longName?: string;
        regularMarketPrice?: number;
        previousClose?: number;
        currency?: string;
        exchangeName?: string;
        regularMarketTime?: number;
      };
    }>;
  };
};

type ShillerLatest = {
  stock_market?: { date?: string; sp500?: number };
};

type QuoteSpec = { symbol: string; label: string; unit: string };

const SHILLER_LATEST_URL = "https://posix4e.github.io/shiller_wrapper_data/data/latest.json";

const RELAYS = [
  (target: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(target)}`,
  (target: string) => `https://corsproxy.io/?${encodeURIComponent(target)}`,
  (target: string) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(target)}`,
];

const TICKER_ALIASES: Record<string, QuoteSpec> = {
  brent: { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  wti: { symbol: "CL=F", label: "WTI Crude", unit: "USD/barrel" },
  oil: { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  gold: { symbol: "GC=F", label: "זהב (GC=F)", unit: "USD/oz" },
  xau: { symbol: "GC=F", label: "זהב (GC=F)", unit: "USD/oz" },
  silver: { symbol: "SI=F", label: "כסף (SI=F)", unit: "USD/oz" },
  sp500: { symbol: "^GSPC", label: "S&P 500", unit: "points" },
  "s&p": { symbol: "^GSPC", label: "S&P 500", unit: "points" },
  nasdaq: { symbol: "^IXIC", label: "NASDAQ Composite", unit: "points" },
  dow: { symbol: "^DJI", label: "Dow Jones", unit: "points" },
  nvidia: { symbol: "NVDA", label: "NVIDIA (NVDA)", unit: "USD" },
  apple: { symbol: "AAPL", label: "Apple (AAPL)", unit: "USD" },
  tesla: { symbol: "TSLA", label: "Tesla (TSLA)", unit: "USD" },
  aapl: { symbol: "AAPL", label: "Apple (AAPL)", unit: "USD" },
  nvda: { symbol: "NVDA", label: "NVIDIA (NVDA)", unit: "USD" },
  tsla: { symbol: "TSLA", label: "Tesla (TSLA)", unit: "USD" },
  חבית: { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  נפט: { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  זהב: { symbol: "GC=F", label: "זהב (GC=F)", unit: "USD/oz" },
};

const pickQuote = (query: string): QuoteSpec => {
  const q = query.toLowerCase();
  if (/wti|west\s+texas/i.test(q)) return TICKER_ALIASES.wti;
  if (/s&p|sp\s*500|sp500|מדד\s*500/i.test(q)) return TICKER_ALIASES.sp500;
  if (/nasdaq|נאסד(?:ק|א)/i.test(q)) return TICKER_ALIASES.nasdaq;
  if (/dow\s*jones|דow|דאו/i.test(q)) return TICKER_ALIASES.dow;
  if (/nvidia|nvda/i.test(q)) return TICKER_ALIASES.nvidia;
  if (/tesla|tsla/i.test(q)) return TICKER_ALIASES.tesla;
  if (/apple|aapl|אפל\b/i.test(q)) return TICKER_ALIASES.apple;
  if (/כסף|silver/i.test(q)) return TICKER_ALIASES.silver;
  if (/זהב|gold|xau|אונקי/i.test(q)) return TICKER_ALIASES.gold;
  if (/brent|חבית|נפט|oil|crude|petroleum/i.test(q)) return TICKER_ALIASES.brent;
  if (/מדד|index/i.test(q)) return TICKER_ALIASES.sp500;
  return TICKER_ALIASES.gold;
};

const yahooChartUrl = (symbol: string): string =>
  `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=1d`;

const parseYahooChart = (data: YahooChart) => {
  const meta = data.chart?.result?.[0]?.meta;
  if (meta?.regularMarketPrice == null || !Number.isFinite(meta.regularMarketPrice)) return null;
  return meta;
};

const fetchYahooViaRelay = async (url: string, timeoutMs: number): Promise<YahooChart | null> => {
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    for (const relay of RELAYS) {
      try {
        const response = await fetch(relay(url), {
          method: "GET",
          signal: controller.signal,
          headers: { Accept: "application/json" },
        });
        if (!response.ok) continue;
        return (await response.json()) as YahooChart;
      } catch {
        /* try next relay */
      }
    }
    return null;
  } finally {
    globalThis.clearTimeout(timer);
  }
};

const fetchYahooQuote = async (spec: QuoteSpec): Promise<{
  price: number;
  prev: number | null;
  when: string;
  symbol: string;
  name: string;
  sourceLabel: string;
} | null> => {
  const url = yahooChartUrl(spec.symbol);
  const timeoutMs = isStaticWebHost() ? 14_000 : 18_000;

  const tasks: Array<Promise<YahooChart | null>> = [
    fetchJson<YahooChart>(url, undefined, { timeoutMs }).catch(() => null),
  ];
  if (isStaticWebHost()) {
    tasks.push(fetchYahooViaRelay(url, timeoutMs));
  }

  const results = await Promise.all(tasks);
  for (const data of results) {
    if (!data) continue;
    const meta = parseYahooChart(data);
    if (!meta) continue;
    return {
      price: meta.regularMarketPrice!,
      prev: meta.previousClose ?? null,
      when:
        meta.regularMarketTime != null
          ? new Date(meta.regularMarketTime * 1000).toISOString().replace("T", " ").slice(0, 19)
          : "—",
      symbol: meta.symbol ?? spec.symbol,
      name: meta.shortName ?? meta.longName ?? spec.label,
      sourceLabel: "Yahoo Finance",
    };
  }

  if (!isStaticWebHost()) return null;

  try {
    const response = await proxyAwareFetch(url, { headers: { Accept: "application/json" } });
    if (response.ok) {
      const data = (await response.json()) as YahooChart;
      const meta = parseYahooChart(data);
      if (meta) {
        return {
          price: meta.regularMarketPrice!,
          prev: meta.previousClose ?? null,
          when:
            meta.regularMarketTime != null
              ? new Date(meta.regularMarketTime * 1000).toISOString().replace("T", " ").slice(0, 19)
              : "—",
          symbol: meta.symbol ?? spec.symbol,
          name: meta.shortName ?? meta.longName ?? spec.label,
          sourceLabel: "Yahoo Finance",
        };
      }
    }
  } catch {
    /* fall through */
  }

  return null;
};

/** Monthly Shiller S&P — CORS-friendly fallback when Yahoo/Stooq fail on static hosts. */
const fetchShillerSp500 = async (): Promise<{ price: number; when: string } | null> => {
  try {
    const data = await fetchJson<ShillerLatest>(SHILLER_LATEST_URL, undefined, { timeoutMs: 12_000 });
    const price = data.stock_market?.sp500;
    if (price == null || !Number.isFinite(price)) return null;
    return { price, when: data.stock_market?.date ?? "—" };
  } catch {
    return null;
  }
};

const fetchStooqQuote = async (spec: QuoteSpec): Promise<{ price: number; when: string } | null> => {
  const stooqSymbols =
    spec.symbol === "^GSPC"
      ? ["^spx", "spx.us", "^spx.us"]
      : [spec.symbol.replace("^", "").toLowerCase()];
  for (const stooqSymbol of stooqSymbols) {
    try {
      const csv = await fetchText(
        `https://stooq.com/q/l/?s=${encodeURIComponent(stooqSymbol)}&f=sd2t2ohlcv&h&e=csv`,
        undefined,
        { timeoutMs: 10_000 },
      );
      const line = csv.trim().split("\n").find((l) => l && !/^symbol,/i.test(l));
      if (!line) continue;
      const cols = line.split(",");
      const close = parseFloat(cols[6] ?? cols[cols.length - 2] ?? "");
      if (!Number.isFinite(close)) continue;
      const date = cols[1] ?? new Date().toISOString().slice(0, 10);
      const time = cols[2] ?? "";
      return { price: close, when: `${date} ${time}`.trim() };
    } catch {
      /* next symbol */
    }
  }
  return null;
};

export const fetchMarketQuoteSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "yahoo-finance" as const;
  const label = "Yahoo Finance — שוק / סחורות";
  const picked = pickQuote(query);
  try {
    let price: number | null = null;
    let prev: number | null = null;
    let when = "—";
    let symbol = picked.symbol;
    let name = picked.label;
    let sourceLabel = "Yahoo Finance";

    if (picked.symbol === "^GSPC" && isStaticWebHost()) {
      const shillerFirst = await fetchShillerSp500();
      if (shillerFirst) {
        price = shillerFirst.price;
        when = shillerFirst.when;
        sourceLabel = "Shiller (monthly)";
      }
    }

    if (price == null) {
      const [yahoo, stooq, shiller] = await Promise.all([
        fetchYahooQuote(picked),
        fetchStooqQuote(picked),
        picked.symbol === "^GSPC" ? fetchShillerSp500() : Promise.resolve(null),
      ]);

      if (yahoo) {
        price = yahoo.price;
        prev = yahoo.prev;
        when = yahoo.when;
        symbol = yahoo.symbol;
        name = yahoo.name;
        sourceLabel = yahoo.sourceLabel;
      } else if (stooq) {
        price = stooq.price;
        when = stooq.when;
        sourceLabel = "Stooq";
      } else if (shiller) {
        price = shiller.price;
        when = shiller.when;
        sourceLabel = "Shiller (monthly)";
      }
    }

    if (price == null || !Number.isFinite(price)) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצא מחיר ב-Yahoo Finance / Stooq / Shiller",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const change = prev != null && Number.isFinite(prev) ? price - prev : null;
    const changePct =
      change != null && prev != null && prev !== 0 ? (change / prev) * 100 : null;

    const staleNote =
      sourceLabel === "Shiller (monthly)" ? ` (עדכון חודשי — ${when})` : "";

    const lines = [
      `${picked.label}: ${price.toFixed(2)} ${picked.unit}${staleNote}`,
      name !== picked.label ? `שם: ${name}` : "",
      change != null
        ? `שינוי מהסגירה הקודמת: ${change >= 0 ? "+" : ""}${change.toFixed(2)}${changePct != null ? ` (${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}%)` : ""}`
        : "",
      `עדכון (${sourceLabel}): ${when} UTC`,
      `סימול: ${symbol}`,
    ].filter(Boolean);

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://finance.yahoo.com/quote/${encodeURIComponent(picked.symbol)}`,
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

/** @deprecated Use fetchMarketQuoteSearch */
export const fetchCommoditySearch = fetchMarketQuoteSearch;
