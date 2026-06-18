import type { Connect } from "vite";
import type { Plugin, PreviewServer, ViteDevServer } from "vite";

const PREFIX = "/api/hf-scanner";
const UPSTREAM = (process.env.HF_SCANNER_UPSTREAM || "http://127.0.0.1:8765").replace(/\/$/, "");

function proxyMiddleware(): Connect.NextHandleFunction {
  return async (req, res, next) => {
    if (!req.url?.startsWith(PREFIX)) {
      next();
      return;
    }
    const suffix = req.url.slice(PREFIX.length) || "/";
    const target = `${UPSTREAM}${suffix}`;
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 30_000);
      const headers: Record<string, string> = {
        Accept: "application/json",
        "Content-Type": req.headers["content-type"] || "application/json",
      };
      if (req.headers.authorization) {
        headers.Authorization = String(req.headers.authorization);
      }
      let body: string | undefined;
      if (req.method && !["GET", "HEAD"].includes(req.method)) {
        body = await new Promise<string>((resolve, reject) => {
          const chunks: Buffer[] = [];
          req.on("data", (c) => chunks.push(Buffer.from(c)));
          req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
          req.on("error", reject);
        });
      }
      const response = await fetch(target, {
        method: req.method || "GET",
        headers,
        body,
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      const text = await response.text();
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Content-Type", response.headers.get("content-type") || "application/json; charset=utf-8");
      res.statusCode = response.status;
      res.end(text);
    } catch (err) {
      res.statusCode = 502;
      res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.end(err instanceof Error ? err.message : "HF scanner proxy failed");
    }
  };
}

function attach(server: ViteDevServer | PreviewServer) {
  server.middlewares.use(proxyMiddleware());
}

/** Dev proxy to local hf-api-scanner (127.0.0.1:8765). */
export function hfScannerProxyPlugin(): Plugin {
  return {
    name: "grovee-hf-scanner-proxy",
    configureServer: attach,
    configurePreviewServer: attach,
  };
}
