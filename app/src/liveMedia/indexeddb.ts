import { openDB, type DBSchema, type IDBPDatabase } from "idb";
import type { Channel, RadioStation, Source } from "./types";
import type { LiveMediaUserPrefs } from "./userPrefs";

interface LiveMediaDB extends DBSchema {
  channels: {
    key: string;
    value: Channel;
    indexes: { "by-source": string; "by-category": string };
  };
  radio: {
    key: string;
    value: RadioStation;
  };
  sources: {
    key: string;
    value: Source;
  };
  userPrefs: {
    key: string;
    value: LiveMediaUserPrefs;
  };
}

let dbPromise: Promise<IDBPDatabase<LiveMediaDB>> | null = null;

export function getLiveMediaDB() {
  if (!dbPromise) {
    dbPromise = openDB<LiveMediaDB>("grovee-live-media", 2, {
      upgrade(db, oldVersion) {
        if (oldVersion < 1) {
          const channels = db.createObjectStore("channels", { keyPath: "id" });
          channels.createIndex("by-source", "source");
          channels.createIndex("by-category", "category");
          db.createObjectStore("radio", { keyPath: "id" });
          db.createObjectStore("sources", { keyPath: "id" });
        }
        if (oldVersion < 2 && !db.objectStoreNames.contains("userPrefs")) {
          db.createObjectStore("userPrefs", { keyPath: "version" });
        }
      },
    });
  }
  return dbPromise;
}

export async function dbPutChannels(channels: Channel[]) {
  const db = await getLiveMediaDB();
  const tx = db.transaction("channels", "readwrite");
  await Promise.all(channels.map((c) => tx.store.put(c)));
  await tx.done;
}

export async function dbGetAllChannels(): Promise<Channel[]> {
  const db = await getLiveMediaDB();
  return db.getAll("channels");
}

export async function dbUpdateChannel(channel: Channel) {
  const db = await getLiveMediaDB();
  await db.put("channels", channel);
}

export async function dbClearChannelsBySource(source: string) {
  const db = await getLiveMediaDB();
  const tx = db.transaction("channels", "readwrite");
  const idx = tx.store.index("by-source");
  let cursor = await idx.openCursor(source);
  while (cursor) {
    await cursor.delete();
    cursor = await cursor.continue();
  }
  await tx.done;
}

export async function dbPutRadio(stations: RadioStation[]) {
  const db = await getLiveMediaDB();
  const tx = db.transaction("radio", "readwrite");
  await Promise.all(stations.map((s) => tx.store.put(s)));
  await tx.done;
}

export async function dbGetAllRadio(): Promise<RadioStation[]> {
  const db = await getLiveMediaDB();
  return db.getAll("radio");
}

export async function dbUpdateRadio(station: RadioStation) {
  const db = await getLiveMediaDB();
  await db.put("radio", station);
}

export async function dbGetAllSources(): Promise<Source[]> {
  const db = await getLiveMediaDB();
  return db.getAll("sources");
}

export async function dbPutSource(source: Source) {
  const db = await getLiveMediaDB();
  await db.put("sources", source);
}

export async function dbGetStats() {
  const db = await getLiveMediaDB();
  const [channels, radio] = await Promise.all([db.count("channels"), db.count("radio")]);
  return { channels, radio };
}

const PREFS_DB_KEY = 2;

export async function dbGetUserPrefs(): Promise<LiveMediaUserPrefs | undefined> {
  const db = await getLiveMediaDB();
  const v2 = await db.get("userPrefs", PREFS_DB_KEY);
  if (v2) return v2;
  return db.get("userPrefs", 1);
}

export async function dbPutUserPrefs(prefs: LiveMediaUserPrefs): Promise<void> {
  const db = await getLiveMediaDB();
  await db.put("userPrefs", { ...prefs, version: PREFS_DB_KEY });
}
