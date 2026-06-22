import { openDB, type DBSchema, type IDBPDatabase } from "idb";
import type { GameCategoryId, OnlineGame } from "../gameSearch/types";

const MAX_PLAYED = 40;
const MAX_FAVORITES = 80;

interface PlayedEntry {
  id: string;
  game: OnlineGame;
  playedAt: number;
  playCount: number;
}

export interface GamesSessionSnapshot {
  games: OnlineGame[];
  title: string;
  category: GameCategoryId | null;
  updatedAt: number;
}

type GamesSessionRecord = GamesSessionSnapshot & { key: string };

interface GroveeExperienceDB extends DBSchema {
  playedGames: {
    key: string;
    value: PlayedEntry;
  };
  favoriteGames: {
    key: string;
    value: OnlineGame;
  };
  gamesSession: {
    key: string;
    value: GamesSessionRecord;
  };
}

let dbPromise: Promise<IDBPDatabase<GroveeExperienceDB>> | null = null;

function getDb() {
  if (!dbPromise) {
    dbPromise = openDB<GroveeExperienceDB>("grovee-experience", 1, {
      upgrade(db) {
        db.createObjectStore("playedGames", { keyPath: "id" });
        db.createObjectStore("favoriteGames", { keyPath: "id" });
        db.createObjectStore("gamesSession", { keyPath: "key" });
      },
    });
  }
  return dbPromise;
}

export async function recordGamePlay(game: OnlineGame): Promise<void> {
  const db = await getDb();
  const prev = await db.get("playedGames", game.id);
  const entry: PlayedEntry = {
    id: game.id,
    game,
    playedAt: Date.now(),
    playCount: (prev?.playCount ?? 0) + 1,
  };
  await db.put("playedGames", entry);
  const all = await db.getAll("playedGames");
  if (all.length > MAX_PLAYED) {
    const sorted = all.sort((a, b) => b.playedAt - a.playedAt);
    const drop = sorted.slice(MAX_PLAYED);
    const tx = db.transaction("playedGames", "readwrite");
    await Promise.all(drop.map((d) => tx.store.delete(d.id)));
    await tx.done;
  }
}

export async function getRecentPlayedGames(limit = 24): Promise<OnlineGame[]> {
  const db = await getDb();
  const all = await db.getAll("playedGames");
  return all
    .sort((a, b) => b.playedAt - a.playedAt)
    .slice(0, limit)
    .map((e) => e.game);
}

export async function getFavoriteGames(): Promise<OnlineGame[]> {
  const db = await getDb();
  const all = await db.getAll("favoriteGames");
  return all.sort((a, b) => a.title.localeCompare(b.title, "he"));
}

export async function isFavoriteGame(id: string): Promise<boolean> {
  const db = await getDb();
  return !!(await db.get("favoriteGames", id));
}

export async function getFavoriteIds(): Promise<Set<string>> {
  const db = await getDb();
  const all = await db.getAllKeys("favoriteGames");
  return new Set(all);
}

export async function toggleFavoriteGame(game: OnlineGame): Promise<boolean> {
  const db = await getDb();
  const existing = await db.get("favoriteGames", game.id);
  if (existing) {
    await db.delete("favoriteGames", game.id);
    return false;
  }
  await db.put("favoriteGames", game);
  const count = await db.count("favoriteGames");
  if (count > MAX_FAVORITES) {
    const all = await db.getAll("favoriteGames");
    all.sort((a, b) => a.title.localeCompare(b.title));
    await db.delete("favoriteGames", all[0].id);
  }
  return true;
}

export async function saveGamesSession(
  games: OnlineGame[],
  title: string,
  category: GameCategoryId | null = null,
): Promise<void> {
  const db = await getDb();
  await db.put("gamesSession", {
    key: "last",
    games,
    title,
    category,
    updatedAt: Date.now(),
  } satisfies GamesSessionRecord);
}

export async function loadGamesSession(): Promise<GamesSessionSnapshot | null> {
  const db = await getDb();
  const row = await db.get("gamesSession", "last");
  if (!row) return null;
  const { key: _k, ...snapshot } = row;
  return snapshot;
}
