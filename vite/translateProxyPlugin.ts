import type { Connect } from "vite";
import type { Plugin, PreviewServer, ViteDevServer } from "vite";

const PROXY_PATH = "/api/translate";

type TranslateBody = {
  texts?: string[];
  target?: string;
  source?: string;
};

function readBody(req: Connect.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function proxyMiddleware(): Connect.NextHandleFunction {
  return async (req, res, next) => {
    if (!req.url?.startsWith(PROXY_PATH) || req.method !== "POST") {
      next();
      return;
    }

    const key = process.env.GOOGLE_TRANSLATE_API_KEY?.trim();
    if (!key) {
      res.statusCode = 503;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify({ error: "GOOGLE_TRANSLATE_API_KEY not set on dev server" }));
      return;
    }

    try {
      const raw = await readBody(req);
      const body = JSON.parse(raw) as TranslateBody;
      const texts = body.texts?.filter((t) => typeof t === "string" && t.trim()) ?? [];
      if (!texts.length) {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "Missing texts" }));
        return;
      }

      const target = body.target || "he";
      const source = body.source || "en";

      const upstream = await fetch(
        `https://translation.googleapis.com/language/translate/v2?key=${encodeURIComponent(key)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ q: texts, target, source, format: "text" }),
        },
      );

      const payload = (await upstream.json()) as {
        data?: { translations?: { translatedText: string }[] };
        error?: { message?: string };
      };

      if (!upstream.ok) {
        res.statusCode = upstream.status;
        res.end(JSON.stringify({ error: payload.error?.message || "Translate upstream failed" }));
        return;
      }

      const translations = payload.data?.translations?.map((t) => t.translatedText) ?? [];

      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.statusCode = 200;
      res.end(JSON.stringify({ translations, provider: "cloud" }));
    } catch (err) {
      res.statusCode = 502;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Translate proxy failed" }));
    }
  };
}

function attach(server: ViteDevServer | PreviewServer) {
  server.middlewares.use(proxyMiddleware());
}

export function translateProxyPlugin(): Plugin {
  return {
    name: "grovee-translate-proxy",
    configureServer: attach,
    configurePreviewServer: attach,
  };
}
