import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FAVORITES_FILE = path.resolve(__dirname, "../public/games/curatedFavorites.json");
const CURATED_GAME_FAVORITES_API_PATH = "/api/games/curated-favorites";

type CuratedGameFavoritesPayload = {
  version?: number;
  description?: string;
  updatedAt?: number;
  games?: unknown[];
};

const readJsonBody = async (req: IncomingMessage): Promise<CuratedGameFavoritesPayload> => {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as CuratedGameFavoritesPayload;
};

const sendJson = (res: ServerResponse, status: number, body: unknown) => {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.end(JSON.stringify(body));
};

const validatePayload = (body: CuratedGameFavoritesPayload): string | null => {
  if (body.version !== 1) return "version must be 1";
  if (!Array.isArray(body.games)) return "games must be an array";
  return null;
};

const writeFavoritesFile = (body: CuratedGameFavoritesPayload): void => {
  fs.mkdirSync(path.dirname(FAVORITES_FILE), { recursive: true });
  const normalized = {
    version: 1,
    description:
      body.description ??
      "Curated game favorites for hero rotation — source of truth in git. Auto-updated in dev when starring ☆.",
    updatedAt: typeof body.updatedAt === "number" ? body.updatedAt : Date.now(),
    games: body.games ?? [],
  };
  const tmp = `${FAVORITES_FILE}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(normalized, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, FAVORITES_FILE);
};

/** Dev-only: read/write public/games/curatedFavorites.json from the UI. */
export function gameFavoritesSyncPlugin(): Plugin {
  return {
    name: "grovee-game-favorites-sync",
    configureServer(server) {
      server.middlewares.use(CURATED_GAME_FAVORITES_API_PATH, async (req, res) => {
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
                "Curated game favorites for hero rotation — source of truth in git. Auto-updated in dev when starring ☆.",
              updatedAt: 0,
              games: [],
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
              path: "public/games/curatedFavorites.json",
              games: body.games?.length ?? 0,
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
