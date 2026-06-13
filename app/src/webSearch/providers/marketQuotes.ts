import { fetchJson } from "../fetchJson";
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

type QuoteSpec = { symbol: string; label: string; unit: string };

const TICKER_ALIASES: Record<string, QuoteSpec> = {
  brent: { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  wti: { symbol: "CL=F", label: "WTI Crude", unit: "USD/barrel" },
  oil: { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  gold: { symbol: "GC=F", label: "זהב (GC=F)", unit: "USD/oz" },
  xau: { symbol: "GC=F", label: "זהב (GC=F)", unit: "USD/oz" },
  silver: { symbol: "SI=F", label: "כסף (SI=F)", unit: "USD/oz" },
  "sp500": { symbol: "^GSPC", label: "S&P 500", unit: "points" },
  "s&p": { symbol: "^GSPC", label: "S&P 500", unit: "points" },
  nasdaq: { symbol: "^IXIC", label: "NASDAQ Composite", unit: "points" },
  dow: { symbol: "^DJI", label: "Dow Jones", unit: "points" },
  nvidia: { symbol: "NVDA", label: "NVIDIA (NVDA)", unit: "USD" },
  apple: { symbol: "AAPL", label: "Apple (AAPL)", unit: "USD" },
  tesla: { symbol: "TSLA", label: "Tesla (TSLA)", unit: "USD" },
  aapl: { symbol: "AAPL", label: "Apple (AAPL)", unit: "USD" },
  nvda: { symbol: "NVDA", label: "NVIDIA (NVDA)", unit: "USD" },
  tsla: { symbol: "TSLA", label: "Tesla (TSLA)", unit: "USD" },
  "חבית": { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  "נפט": { symbol: "BZ=F", label: "Brent Crude (חבית)", unit: "USD/barrel" },
  "זהב": { symbol: "GC=F", label: "זהב (GC=F)", unit: "USD/oz" },
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

const fetchYahooQuote = async (spec: QuoteSpec): Promise<YahooChart> => {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(spec.symbol)}?interval=1d&range=1d`;
  return fetchJson<YahooChart>(url, undefined, { timeoutMs: 18_000 });
};

export const fetchMarketQuoteSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "yahoo-finance" as const;
  const label = "Yahoo Finance — שוק / סחורות";
  const picked = pickQuote(query);
  try {
    const data = await fetchYahooQuote(picked);
    const meta = data.chart?.result?.[0]?.meta;
    const price = meta?.regularMarketPrice;
    if (price == null || !Number.isFinite(price)) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצא מחיר ב-Yahoo Finance",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const prev = meta?.previousClose;
    const change =
      prev != null && Number.isFinite(prev) ? price - prev : null;
    const changePct =
      change != null && prev != null && prev !== 0 ? (change / prev) * 100 : null;
    const when =
      meta?.regularMarketTime != null
        ? new Date(meta.regularMarketTime * 1000).toISOString().replace("T", " ").slice(0, 19)
        : "—";

    const lines = [
      `${picked.label}: ${price.toFixed(2)} ${picked.unit}`,
      meta?.shortName || meta?.longName ? `שם: ${meta.shortName ?? meta.longName}` : "",
      change != null
        ? `שינוי מהסגירה הקודמת: ${change >= 0 ? "+" : ""}${change.toFixed(2)}${changePct != null ? ` (${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}%)` : ""}`
        : "",
      `עדכון (Yahoo Finance): ${when} UTC`,
      `סימול: ${meta?.symbol ?? picked.symbol}`,
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
