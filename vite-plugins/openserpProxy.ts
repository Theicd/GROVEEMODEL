import type { ServerResponse } from "node:http";
import http from "node:http";
import type { Plugin } from "vite";

const TARGET = (process.env.OPENSERP_URL ?? process.env.VITE_OPENSERP_URL ?? "http://127.0.0.1:7000")
  .trim()
  .replace(/\/$/, "");

const corsHeaders = (res: ServerResponse) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Accept");
};

/** Dev/preview proxy — browser → /api/openserp → local OpenSERP on :7000 */
export function openserpProxyPlugin(): Plugin {
  return {
    name: "grovee-openserp-proxy",
    configureServer(server) {
      server.middlewares.use("/api/openserp", (req, res) => {
        corsHeaders(res);
        if (req.method === "OPTIONS") {
          res.statusCode = 204;
          res.end();
          return;
        }

        const rawPath = req.url ?? "/";
        const path = rawPath.startsWith("/api/openserp")
          ? rawPath.slice("/api/openserp".length) || "/"
          : rawPath;
        const targetUrl = `${TARGET}${path.startsWith("/") ? path : `/${path}`}`;

        const proxyReq = http.request(
          targetUrl,
          {
            method: req.method,
            headers: {
              Accept: req.headers.accept ?? "application/json",
            },
          },
          (proxyRes) => {
            res.statusCode = proxyRes.statusCode ?? 502;
            const ct = proxyRes.headers["content-type"];
            if (ct) res.setHeader("Content-Type", ct);
            corsHeaders(res);
            proxyRes.pipe(res);
          },
        );

        proxyReq.on("error", (err) => {
          res.statusCode = 502;
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          corsHeaders(res);
          res.end(
            JSON.stringify({
              ok: false,
              error: `OpenSERP offline at ${TARGET}: ${err.message}`,
            }),
          );
        });

        if (req.method === "GET" || req.method === "HEAD") {
          proxyReq.end();
          return;
        }

        req.pipe(proxyReq);
      });
    },
  };
}
