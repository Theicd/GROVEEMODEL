import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REALITY_UI = path.resolve(__dirname, "../../reality-core/ui");

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function sendFile(res: ServerResponse, filePath: string) {
  const ext = path.extname(filePath).toLowerCase();
  res.setHeader("Content-Type", MIME[ext] ?? "application/octet-stream");
  fs.createReadStream(filePath).pipe(res);
}

const REALITY_SERVER = process.env.REALITY_SERVER ?? "http://127.0.0.1:3000";
const SEARXNG_UPSTREAM = (process.env.SEARXNG_UPSTREAM ?? process.env.VITE_SEARXNG_UPSTREAM ?? "").replace(/\/$/, "");
const CHEAPERSAL_API_KEY = process.env.CHEAPERSAL_API_KEY ?? process.env.VITE_CHEAPERSAL_API_KEY ?? "";
const CHEAPERSAL_UPSTREAM = (process.env.CHEAPERSAL_UPSTREAM ?? "https://api.cheapersal.co.il/api/v1").replace(/\/$/, "");

/** Dev-only: CORS proxy + serve reality-core UI at /reality/ */
export function devRealityPlugin(): Plugin {
  return {
    name: "grovee-reality-dev",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const raw = req.url ?? "";
        const pathOnly = raw.split("?")[0];
        const isRealityApi =
          pathOnly === "/api/data" ||
          pathOnly === "/api/status" ||
          pathOnly.startsWith("/api/alerts");
        if (!isRealityApi) return next();
        try {
          const upstream = await fetch(`${REALITY_SERVER}${raw}`);
          res.statusCode = upstream.status;
          res.setHeader("Access-Control-Allow-Origin", "*");
          const ct = upstream.headers.get("content-type");
          if (ct) res.setHeader("Content-Type", ct);
          res.end(await upstream.text());
        } catch {
          res.statusCode = 502;
          res.end("Reality Core server unavailable (run npm start in reality-core)");
        }
      });

      server.middlewares.use("/api/searxng", async (req, res) => {
        if (req.method !== "GET") {
          res.statusCode = 405;
          res.end("Method not allowed");
          return;
        }
        if (!SEARXNG_UPSTREAM) {
          res.statusCode = 503;
          res.setHeader("Content-Type", "text/plain; charset=utf-8");
          res.end("SEARXNG_UPSTREAM not set — add to app/.env (see .env.example)");
          return;
        }
        const raw = req.url ?? "/search";
        const target = `${SEARXNG_UPSTREAM}${raw.startsWith("/") ? raw : `/${raw}`}`;
        try {
          const upstream = await fetch(target, {
            headers: {
              Accept: "application/json",
              "User-Agent": "GROVEEMODEL/1.0 (searxng dev proxy)",
            },
          });
          res.statusCode = upstream.status;
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Content-Type", upstream.headers.get("content-type") ?? "application/json");
          res.end(await upstream.text());
        } catch (e) {
          res.statusCode = 502;
          res.end(e instanceof Error ? e.message : "searxng proxy error");
        }
      });

      server.middlewares.use("/api/cheapersal", async (req, res) => {
        if (req.method !== "GET") {
          res.statusCode = 405;
          res.end("Method not allowed");
          return;
        }
        if (!CHEAPERSAL_API_KEY) {
          res.statusCode = 503;
          res.setHeader("Content-Type", "text/plain; charset=utf-8");
          res.end("CHEAPERSAL_API_KEY not set — add to app/.env (see .env.example)");
          return;
        }
        const raw = req.url ?? "/";
        const target = `${CHEAPERSAL_UPSTREAM}${raw.startsWith("/") ? raw : `/${raw}`}`;
        try {
          const upstream = await fetch(target, {
            headers: {
              Accept: "application/json",
              "X-API-Key": CHEAPERSAL_API_KEY,
              "User-Agent": "GROVEEMODEL/1.0 (cheapersal dev proxy)",
            },
          });
          res.statusCode = upstream.status;
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Content-Type", upstream.headers.get("content-type") ?? "application/json");
          res.end(await upstream.text());
        } catch (e) {
          res.statusCode = 502;
          res.end(e instanceof Error ? e.message : "cheapersal proxy error");
        }
      });

      server.middlewares.use("/api/proxy", async (req, res) => {
        if (req.method !== "GET" && req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method not allowed");
          return;
        }
        const u = new URL(req.url ?? "", "http://localhost");
        const target = u.searchParams.get("url");
        if (!target) {
          res.statusCode = 400;
          res.end("Missing url param");
          return;
        }
        try {
          const headers: Record<string, string> = {
            Accept: "application/json, text/plain, */*",
            "User-Agent": "GROVEEMODEL/1.0 (dev proxy)",
          };
          const init: RequestInit = { method: req.method, headers };
          if (req.method === "POST") {
            const chunks: Buffer[] = [];
            for await (const c of req) chunks.push(c as Buffer);
            init.body = Buffer.concat(chunks);
          }
          const upstream = await fetch(target, init);
          res.statusCode = upstream.status;
          res.setHeader("Access-Control-Allow-Origin", "*");
          const ct = upstream.headers.get("content-type");
          if (ct) res.setHeader("Content-Type", ct);
          const body = await upstream.text();
          res.end(body);
        } catch (e) {
          res.statusCode = 502;
          res.end(e instanceof Error ? e.message : "proxy error");
        }
      });

      server.middlewares.use("/reality", (req, res, next) => {
        if (!fs.existsSync(REALITY_UI)) return next();
        let p = (req as IncomingMessage).url?.split("?")[0] ?? "/";
        if (p === "/" || p.endsWith("/")) p = "/israel.html";
        const file = path.normalize(path.join(REALITY_UI, p.replace(/^\//, "")));
        if (!file.startsWith(REALITY_UI) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
          return next();
        }
        sendFile(res, file);
      });
    },
  };
}
