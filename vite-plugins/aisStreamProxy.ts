import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";

const ENV_KEY = (process.env.AISSTREAM_API_KEY ?? process.env.VITE_AISSTREAM_API_KEY ?? "").trim();

type BboxBody = {
  apiKey?: string;
  minLat?: number;
  maxLat?: number;
  minLon?: number;
  maxLon?: number;
  timeoutMs?: number;
};

type AisStreamMessage = {
  error?: string;
  MessageType?: string;
  Metadata?: { MMSI?: number; ShipName?: string; latitude?: number; longitude?: number };
  Message?: {
    PositionReport?: {
      Latitude?: number;
      Longitude?: number;
      Sog?: number;
      UserID?: number;
    };
    ShipStaticData?: { Name?: string; Destination?: string; UserID?: number };
  };
};

export type AisStreamShipRow = {
  name: string;
  lat: number;
  lon: number;
  speed?: number;
  mmsi?: number;
  destination?: string;
  source: "aisstream";
};

const readJsonBody = async (req: IncomingMessage): Promise<BboxBody> => {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as BboxBody;
};

const sendJson = (res: ServerResponse, status: number, body: unknown) => {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.end(JSON.stringify(body));
};

/** Multi-region boxes — major shipping lanes (AISStream subscription filters). */
export const AISSTREAM_GLOBE_BOXES: [number, number][][] = [
  [[30, 18], [42, 38]],
  [[35, -12], [62, 15]],
  [[24, -82], [46, -65]],
  [[12, 32], [31, 44]],
  [[50, -8], [59, 2]],
  [[12, 38], [30, 55]],
  [[1, 95], [22, 110]],
  [[-5, 95], [25, 125]],
  [[20, 115], [42, 130]],
  [[33, 125], [46, 145]],
  [[-35, 15], [5, 45]],
  [[45, -130], [62, -125]],
];

const ingestAisMessage = (
  m: AisStreamMessage,
  byKey: Map<string, AisStreamShipRow>,
  wsErr: { current?: string },
) => {
  if (m.error) {
    wsErr.current = m.error;
    return;
  }
  const meta = m.Metadata;
  const pr = m.Message?.PositionReport;
  const sd = m.Message?.ShipStaticData;
  const mmsi = meta?.MMSI ?? pr?.UserID ?? sd?.UserID;
  const key = mmsi != null ? `mmsi:${mmsi}` : pr ? `pos:${pr.Latitude?.toFixed(3)}:${pr.Longitude?.toFixed(3)}` : null;
  if (!key) return;

  const prev = byKey.get(key);
  if (pr?.Latitude != null && pr?.Longitude != null) {
    const name =
      meta?.ShipName?.trim() ||
      sd?.Name?.trim() ||
      prev?.name ||
      (mmsi != null ? `MMSI ${mmsi}` : "AIS");
    byKey.set(key, {
      name,
      lat: pr.Latitude,
      lon: pr.Longitude,
      speed: pr.Sog,
      mmsi: mmsi ?? undefined,
      destination: sd?.Destination?.trim() || prev?.destination,
      source: "aisstream",
    });
  } else if (sd?.Name && prev) {
    byKey.set(key, { ...prev, name: sd.Name.trim(), destination: sd.Destination?.trim() || prev.destination });
  }
};

const collectAisStreamWithBoxes = async (
  apiKey: string,
  boundingBoxes: [number, number][][],
  timeoutMs = 10_000,
  maxShips = 400,
): Promise<{ ships: AisStreamShipRow[]; error?: string }> => {
  const WebSocketImpl =
    typeof WebSocket !== "undefined"
      ? WebSocket
      : ((await import("ws")).default as unknown as typeof WebSocket);

  return new Promise((resolve) => {
    const byKey = new Map<string, AisStreamShipRow>();
    let settled = false;
    const wsErr: { current?: string } = {};

    const finish = () => {
      if (settled) return;
      settled = true;
      try {
        ws.close();
      } catch {
        /* ignore */
      }
      resolve({ ships: [...byKey.values()].slice(0, maxShips), error: wsErr.current });
    };

    const timer = setTimeout(finish, Math.min(Math.max(timeoutMs, 3000), 20_000));

    const onWs = (
      wsSocket: {
        addEventListener?: (type: string, fn: (ev: { data?: unknown }) => void) => void;
        on?: (event: string, fn: (...args: unknown[]) => void) => void;
        send: (data: string) => void;
        close: () => void;
      },
      event: "open" | "message" | "error" | "close",
      fn: (data?: unknown) => void,
    ) => {
      if (wsSocket.addEventListener) {
        wsSocket.addEventListener(event, (ev) => fn(ev.data));
      } else if (wsSocket.on) {
        wsSocket.on(event, (...args: unknown[]) => fn(args[0]));
      }
    };

    let ws: {
      send: (data: string) => void;
      close: () => void;
      addEventListener?: (type: string, fn: (ev: { data?: unknown }) => void) => void;
      on?: (event: string, fn: (...args: unknown[]) => void) => void;
    };
    try {
      ws = new WebSocketImpl("wss://stream.aisstream.io/v0/stream") as typeof ws;
    } catch (e) {
      clearTimeout(timer);
      resolve({ ships: [], error: e instanceof Error ? e.message : "WebSocket failed" });
      return;
    }

    onWs(ws, "open", () => {
      ws.send(
        JSON.stringify({
          APIKey: apiKey,
          BoundingBoxes: boundingBoxes,
          FilterMessageTypes: ["PositionReport", "ShipStaticData"],
        }),
      );
    });

    onWs(ws, "message", (raw) => {
      try {
        ingestAisMessage(JSON.parse(String(raw)) as AisStreamMessage, byKey, wsErr);
      } catch {
        /* skip malformed frame */
      }
    });

    onWs(ws, "error", () => {
      wsErr.current = wsErr.current ?? "AISStream WebSocket error";
    });

    onWs(ws, "close", () => {
      clearTimeout(timer);
      finish();
    });
  });
};

/** Collect AIS positions via WebSocket (server-side — key never exposed to aisstream.io from browser). */
export const collectAisStreamShips = async (
  apiKey: string,
  minLat: number,
  maxLat: number,
  minLon: number,
  maxLon: number,
  timeoutMs = 10_000,
): Promise<{ ships: AisStreamShipRow[]; error?: string }> =>
  collectAisStreamWithBoxes(
    apiKey,
    [
      [
        [minLat, minLon],
        [maxLat, maxLon],
      ],
    ],
    timeoutMs,
    120,
  );

export const collectAisStreamGlobe = async (
  apiKey: string,
  timeoutMs = 22_000,
): Promise<{ ships: AisStreamShipRow[]; error?: string }> =>
  collectAisStreamWithBoxes(apiKey, AISSTREAM_GLOBE_BOXES, timeoutMs, 1500);

/** Dev/preview: POST /api/aisstream/ships — bbox + optional apiKey in body. */
export function aisStreamProxyPlugin(): Plugin {
  return {
    name: "grovee-aisstream-proxy",
    configureServer(server) {
      server.middlewares.use("/api/aisstream/ships", async (req, res) => {
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
          const apiKey = (body.apiKey ?? ENV_KEY).trim();
          if (!apiKey) {
            sendJson(res, 503, {
              ok: false,
              error: "חסר מפתח AISStream — הוסף במסך מפתחות API או AISSTREAM_API_KEY ב-.env",
            });
            return;
          }
          const { minLat, maxLat, minLon, maxLon } = body;
          if (
            minLat == null ||
            maxLat == null ||
            minLon == null ||
            maxLon == null ||
            !Number.isFinite(minLat + maxLat + minLon + maxLon)
          ) {
            sendJson(res, 400, { ok: false, error: "Missing bbox (minLat, maxLat, minLon, maxLon)" });
            return;
          }
          const { ships, error } = await collectAisStreamShips(
            apiKey,
            minLat,
            maxLat,
            minLon,
            maxLon,
            body.timeoutMs ?? 10_000,
          );
          sendJson(res, 200, {
            ok: true,
            ships,
            count: ships.length,
            fetchedAt: new Date().toISOString(),
            warning: error,
          });
        } catch (e) {
          sendJson(res, 502, { ok: false, error: e instanceof Error ? e.message : "AISStream proxy error" });
        }
      });

      server.middlewares.use("/api/aisstream/globe", async (req, res) => {
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
          const apiKey = (body.apiKey ?? ENV_KEY).trim();
          if (!apiKey) {
            sendJson(res, 503, {
              ok: false,
              error: "חסר מפתח AISStream — הוסף במסך מפתחות API",
            });
            return;
          }
          const { ships, error } = await collectAisStreamGlobe(apiKey, body.timeoutMs ?? 14_000);
          sendJson(res, 200, {
            ok: true,
            ships,
            count: ships.length,
            fetchedAt: new Date().toISOString(),
            regions: AISSTREAM_GLOBE_BOXES.length,
            warning: error,
          });
        } catch (e) {
          sendJson(res, 502, { ok: false, error: e instanceof Error ? e.message : "AISStream globe error" });
        }
      });

      server.middlewares.use("/api/ships/diagnostics", async (req, res) => {
        if (req.method === "OPTIONS") {
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
          res.statusCode = 204;
          res.end();
          return;
        }
        if (req.method !== "GET") {
          sendJson(res, 405, { ok: false, error: "GET only" });
          return;
        }
        try {
          const { runShipPipelineDiagnostics } = await import(
            "../app/src/realityData/shipPipelineDiagnostics.ts"
          );
          const host = req.headers.host ?? "127.0.0.1:5180";
          const report = await runShipPipelineDiagnostics({
            aisStreamKey: ENV_KEY,
            devOrigin: `http://${host}`,
          });
          sendJson(res, 200, { ok: true, ...report });
        } catch (e) {
          sendJson(res, 502, { ok: false, error: e instanceof Error ? e.message : "diagnostics error" });
        }
      });
    },
  };
}
