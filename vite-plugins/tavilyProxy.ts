import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";
import { tavily } from "@tavily/core";

const ENV_KEY = (process.env.TAVILY_API_KEY ?? process.env.VITE_TAVILY_API_KEY ?? "").trim();

export type TavilyWebHit = {
  id: string;
  title: string;
  url: string;
  snippet: string;
  engine?: string;
  score?: number;
};

type SearchBody = {
  query?: string;
  apiKey?: string;
  searchDepth?: "basic" | "advanced" | "fast" | "ultra-fast";
  maxResults?: number;
  topic?: "general" | "news" | "finance";
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

export const mapTavilyResultsToWebHits = (
  results: Array<{ title?: string; url?: string; content?: string; score?: number }>,
  prefix = "tavily",
): TavilyWebHit[] =>
  results
    .filter((r) => r.url?.trim())
    .map((r, i) => ({
      id: `${prefix}-${i}-${(r.url ?? "").slice(0, 48)}`,
      title: (r.title ?? "ללא כותרת").trim(),
      url: r.url!.trim(),
      snippet: (r.content ?? "").replace(/\s+/g, " ").trim().slice(0, 320),
      engine: "Tavily",
      score: r.score,
    }));

export const runTavilySearch = async (input: {
  query: string;
  apiKey: string;
  searchDepth?: SearchBody["searchDepth"];
  maxResults?: number;
  topic?: SearchBody["topic"];
}): Promise<{ hits: TavilyWebHit[]; answer?: string; responseTime?: number }> => {
  const client = tavily({ apiKey: input.apiKey, clientSource: "groveemodel-dev" });
  const response = await client.search(input.query.trim(), {
    searchDepth: input.searchDepth ?? "advanced",
    maxResults: input.maxResults ?? 12,
    topic: input.topic ?? "general",
    includeAnswer: false,
    timeout: 25,
  });
  return {
    hits: mapTavilyResultsToWebHits(response.results ?? []),
    answer: response.answer,
    responseTime: response.responseTime,
  };
};

export function tavilyProxyPlugin(): Plugin {
  return {
    name: "grovee-tavily-proxy",
    configureServer(server) {
      server.middlewares.use("/api/tavily/search", async (req, res) => {
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
              error: "חסר מפתח Tavily — הוסף במסך מפתחות API או TAVILY_API_KEY ב-.env",
            });
            return;
          }
          if (!query) {
            sendJson(res, 400, { ok: false, error: "חסר query" });
            return;
          }
          const { hits, answer, responseTime } = await runTavilySearch({
            query,
            apiKey,
            searchDepth: body.searchDepth,
            maxResults: body.maxResults,
            topic: body.topic,
          });
          sendJson(res, 200, {
            ok: true,
            query,
            hits,
            count: hits.length,
            answer,
            responseTime,
            fetchedAt: new Date().toISOString(),
          });
        } catch (e) {
          sendJson(res, 502, {
            ok: false,
            error: e instanceof Error ? e.message : "Tavily proxy error",
          });
        }
      });
    },
  };
}
