import type { Connect } from "vite";
import type { Plugin } from "vite";

const ALLOWED_EPG_HOSTS = new Set([
  "i.mjh.nz",
  "epg.pw",
  "iptv-org.github.io",
  "raw.githubusercontent.com",
]);

function attachEpgProxy(middlewares: Connect.Server) {
  middlewares.use("/api/epg/raw", async (req, res) => {
    if (req.method !== "GET") {
      res.statusCode = 405;
      res.end("Method not allowed");
      return;
    }
    const u = new URL(req.url ?? "", "http://127.0.0.1");
    const target = u.searchParams.get("url");
    if (!target) {
      res.statusCode = 400;
      res.end("Missing url param");
      return;
    }
    let parsed: URL;
    try {
      parsed = new URL(target);
    } catch {
      res.statusCode = 400;
      res.end("Invalid url");
      return;
    }
    if (!ALLOWED_EPG_HOSTS.has(parsed.hostname) || parsed.protocol !== "https:") {
      res.statusCode = 403;
      res.end(`EPG proxy host not allowed: ${parsed.hostname}`);
      return;
    }
    try {
      const upstream = await fetch(target, {
        headers: { Accept: "application/octet-stream, */*", "User-Agent": "GROVEEMODEL/1.0 (epg proxy)" },
      });
      res.statusCode = upstream.status;
      res.setHeader("Access-Control-Allow-Origin", "*");
      const ct = upstream.headers.get("content-type");
      if (ct) res.setHeader("Content-Type", ct);
      const buf = Buffer.from(await upstream.arrayBuffer());
      res.end(buf);
    } catch (e) {
      res.statusCode = 502;
      res.end(e instanceof Error ? e.message : "EPG proxy error");
    }
  });
}

/** Dev/preview: same-origin binary proxy for MJH XMLTV (.gz) — /api/proxy corrupts gzip via .text(). */
export function mjhEpgProxyPlugin(): Plugin {
  return {
    name: "grovee-mjh-epg-proxy",
    configureServer(server) {
      attachEpgProxy(server.middlewares);
    },
    configurePreviewServer(server) {
      attachEpgProxy(server.middlewares);
    },
  };
}
