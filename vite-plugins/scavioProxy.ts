import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";

const ENV_KEY = (process.env.SCAVIO_API_KEY ?? process.env.VITE_SCAVIO_API_KEY ?? "").trim();
const SCAVIO_URL = "https://api.scavio.dev/api/v1/google";

export type ScavioWebHit = {
  id: string;
  title: string;
  url: string;
  snippet: string;
  engine?: string;
  position?: number;
};

type SearchBody = {
  query?: string;
  apiKey?: string;
  lightRequest?: boolean;
  countryCode?: string;
  language?: string;
  searchType?: string;
  maxResults?: number;
};

const readJsonBody = async (req: IncomingMessage): Promise<SearchBody> => {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as SearchBody;
};

const sendJson = (res: ServerResponse, status: number, body: unknown) => {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.end(JSON.stringify(body));
};

export type ScavioResultRow = {
  title: string;
  url: string;
  content: string;
  position?: number;
};

/** Parse Scavio Google JSON — supports docs `results[]` and legacy `organic_results[]`. */
export const parseScavioGoogleResponse = (
  raw: unknown,
): { results: ScavioResultRow[]; query?: string; error?: string; creditsRemaining?: number } => {
  if (!raw || typeof raw !== "object") return { results: [], error: "empty response" };
  const o = raw as Record<string, unknown>;
  if (o.error) return { results: [], error: String(o.error) };
  if (typeof o.message === "string" && !Array.isArray(o.results) && !Array.isArray(o.organic_results)) {
    return { results: [], error: o.message };
  }

  const rows = (o.results ?? o.organic_results) as unknown;
  if (!Array.isArray(rows)) return { results: [], error: "no results array" };

  const results: ScavioResultRow[] = [];
  for (const item of rows) {
    if (!item || typeof item !== "object") continue;
    const r = item as Record<string, unknown>;
    const url = String(r.url ?? r.link ?? "").trim();
    if (!url) continue;
    results.push({
      title: String(r.title ?? "ללא כותרת").trim(),
      url,
      content: String(r.content ?? r.snippet ?? "").replace(/\s+/g, " ").trim(),
      position: typeof r.position === "number" ? r.position : undefined,
    });
  }

  return {
    results,
    query: typeof o.query === "string" ? o.query : undefined,
    creditsRemaining: typeof o.credits_remaining === "number" ? o.credits_remaining : undefined,
  };
};

export const mapScavioResultsToWebHits = (
  results: ScavioResultRow[],
  prefix = "scavio",
): ScavioWebHit[] =>
  results.map((r, i) => ({
    id: `${prefix}-${i}-${r.url.slice(0, 48)}`,
    title: r.title,
    url: r.url,
    snippet: r.content.slice(0, 320),
    engine: "Scavio Google",
    position: r.position,
  }));

export const runScavioGoogleSearch = async (input: {
  query: string;
  apiKey: string;
  lightRequest?: boolean;
  countryCode?: string;
  language?: string;
  searchType?: string;
  maxResults?: number;
}): Promise<{ hits: ScavioWebHit[]; query?: string; creditsRemaining?: number; error?: string }> => {
  const body: Record<string, unknown> = {
    query: input.query.trim(),
    light_request: input.lightRequest ?? true,
  };
  if (input.countryCode) body.country_code = input.countryCode;
  if (input.language) body.language = input.language;
  if (input.searchType) body.search_type = input.searchType;

  const res = await fetch(SCAVIO_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${input.apiKey}`,
      "Content-Type": "application/json",
      Accept: "application/json",
      "User-Agent": "GROVEEMODEL/1.0 (dev scavio proxy)",
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(28_000),
  });

  const raw = await res.json();
  if (!res.ok) {
    const err =
      typeof raw === "object" && raw && "message" in raw
        ? String((raw as { message?: string }).message)
        : `HTTP ${res.status}`;
    return { hits: [], error: err };
  }

  const { results, query, creditsRemaining, error } = parseScavioGoogleResponse(raw);
  const capped = input.maxResults ? results.slice(0, input.maxResults) : results.slice(0, 12);
  return {
    hits: mapScavioResultsToWebHits(capped),
    query,
    creditsRemaining,
    error: error && !capped.length ? error : undefined,
  };
};

export function scavioProxyPlugin(): Plugin {
  return {
    name: "grovee-scavio-proxy",
    configureServer(server) {
      server.middlewares.use("/api/scavio/google", async (req, res) => {
        if (req.method === "OPTIONS") {
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
          res.setHeader("Access-Control-Allow-Headers", "Content-Type");
          res.statusCode = 204;
          res.end();
          return;
        }
        if (req.method !== "POST") {
          sendJson(res, 405, { ok: false, error: "POST only" });
          return;
        }
        try {
          const body = await readJsonBody(req);
          const query = (body.query ?? "").trim();
          const apiKey = (body.apiKey ?? ENV_KEY).trim();
          if (!apiKey) {
            sendJson(res, 503, {
              ok: false,
              error: "חסר מפתח Scavio — הוסף במסך מפתחות API או SCAVIO_API_KEY ב-.env",
            });
            return;
          }
          if (!query) {
            sendJson(res, 400, { ok: false, error: "חסר query" });
            return;
          }
          const { hits, query: q, creditsRemaining, error } = await runScavioGoogleSearch({
            query,
            apiKey,
            lightRequest: body.lightRequest,
            countryCode: body.countryCode,
            language: body.language,
            searchType: body.searchType,
            maxResults: body.maxResults,
          });
          if (error && !hits.length) {
            sendJson(res, 502, { ok: false, error });
            return;
          }
          sendJson(res, 200, {
            ok: true,
            query: q ?? query,
            hits,
            count: hits.length,
            creditsRemaining,
            fetchedAt: new Date().toISOString(),
          });
        } catch (e) {
          sendJson(res, 502, { ok: false, error: e instanceof Error ? e.message : "Scavio proxy error" });
        }
      });
    },
  };
}
