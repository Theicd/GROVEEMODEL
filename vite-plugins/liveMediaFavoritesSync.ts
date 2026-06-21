import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FAVORITES_FILE = path.resolve(__dirname, "../public/liveMedia/curatedFavorites.json");
const CURATED_FAVORITES_API_PATH = "/api/live-media/curated-favorites";

type CuratedFavoritesPayload = {
  version?: number;
  description?: string;
  updatedAt?: number;
  channels?: unknown[];
  radio?: unknown[];
};

const readJsonBody = async (req: IncomingMessage): Promise<CuratedFavoritesPayload> => {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as CuratedFavoritesPayload;
};

const sendJson = (res: ServerResponse, status: number, body: unknown) => {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.end(JSON.stringify(body));
};

const validatePayload = (body: CuratedFavoritesPayload): string | null => {
  if (body.version !== 1) return "version must be 1";
  if (!Array.isArray(body.channels)) return "channels must be an array";
  if (!Array.isArray(body.radio)) return "radio must be an array";
  return null;
};

const writeFavoritesFile = (body: CuratedFavoritesPayload): void => {
  fs.mkdirSync(path.dirname(FAVORITES_FILE), { recursive: true });
  const normalized = {
    version: 1,
    description:
      body.description ??
      "Curated TV/radio favorites — source of truth in git. Auto-updated when starring in dev (npm run dev).",
    updatedAt: typeof body.updatedAt === "number" ? body.updatedAt : Date.now(),
    channels: body.channels ?? [],
    radio: body.radio ?? [],
  };
  const tmp = `${FAVORITES_FILE}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(normalized, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, FAVORITES_FILE);
};

/** Dev-only: read/write public/liveMedia/curatedFavorites.json from the UI. */
export function liveMediaFavoritesSyncPlugin(): Plugin {
  return {
    name: "grovee-live-media-favorites-sync",
    configureServer(server) {
      server.middlewares.use(CURATED_FAVORITES_API_PATH, async (req, res) => {
        if (req.method === "OPTIONS") {
          res.statusCode = 204;
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
          res.setHeader("Access-Control-Allow-Headers", "Content-Type");
          res.end();
          return;
        }

        if (req.method === "GET") {
          if (!fs.existsSync(FAVORITES_FILE)) {
            sendJson(res, 200, {
              version: 1,
              description:
                "Curated TV/radio favorites — source of truth in git. Auto-updated when starring in dev (npm run dev).",
              updatedAt: 0,
              channels: [],
              radio: [],
            });
            return;
          }
          const raw = fs.readFileSync(FAVORITES_FILE, "utf8");
          sendJson(res, 200, JSON.parse(raw));
          return;
        }

        if (req.method === "POST") {
          try {
            const body = await readJsonBody(req);
            const err = validatePayload(body);
            if (err) {
              sendJson(res, 400, { ok: false, error: err });
              return;
            }
            writeFavoritesFile(body);
            sendJson(res, 200, {
              ok: true,
              path: "public/liveMedia/curatedFavorites.json",
              channels: body.channels?.length ?? 0,
              radio: body.radio?.length ?? 0,
            });
          } catch (e) {
            sendJson(res, 500, {
              ok: false,
              error: e instanceof Error ? e.message : String(e),
            });
          }
          return;
        }

        sendJson(res, 405, { ok: false, error: "Method not allowed" });
      });
    },
  };
}
