import { fetchJson, fetchText } from "../fetchJson";
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

const fetchStooqQuote = async (spec: QuoteSpec): Promise<{ price: number; when: string } | null> => {
  const stooqSymbol =
    spec.symbol === "^GSPC" ? "^spx" : spec.symbol.replace("^", "").toLowerCase();
  const csv = await fetchText(
    `https://stooq.com/q/l/?s=${encodeURIComponent(stooqSymbol)}&f=sd2t2ohlcv&h&e=csv`,
    undefined,
    { timeoutMs: 16_000 },
  );
  const line = csv.trim().split("\n").find((l) => l && !/^symbol,/i.test(l));
  if (!line) return null;
  const cols = line.split(",");
  const close = parseFloat(cols[6] ?? cols[cols.length - 2] ?? "");
  if (!Number.isFinite(close)) return null;
  const date = cols[1] ?? new Date().toISOString().slice(0, 10);
  const time = cols[2] ?? "";
  return { price: close, when: `${date} ${time}`.trim() };
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

    try {
      const data = await fetchYahooQuote(picked);
      const meta = data.chart?.result?.[0]?.meta;
      if (meta?.regularMarketPrice != null && Number.isFinite(meta.regularMarketPrice)) {
        price = meta.regularMarketPrice;
        prev = meta.previousClose ?? null;
        symbol = meta.symbol ?? picked.symbol;
        name = meta.shortName ?? meta.longName ?? picked.label;
        when =
          meta.regularMarketTime != null
            ? new Date(meta.regularMarketTime * 1000).toISOString().replace("T", " ").slice(0, 19)
            : "—";
      }
    } catch {
      /* stooq fallback */
    }

    if (price == null) {
      const stooq = await fetchStooqQuote(picked);
      if (stooq) {
        price = stooq.price;
        when = stooq.when;
        sourceLabel = "Stooq";
      }
    }

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

    const change =
      prev != null && Number.isFinite(prev) ? price - prev : null;
    const changePct =
      change != null && prev != null && prev !== 0 ? (change / prev) * 100 : null;

    const lines = [
      `${picked.label}: ${price.toFixed(2)} ${picked.unit}`,
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
