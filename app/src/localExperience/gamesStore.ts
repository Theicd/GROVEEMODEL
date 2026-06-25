import { openDB, type DBSchema, type IDBPDatabase } from "idb";
import type { GameCategoryId, OnlineGame } from "../gameSearch/types";
import {
  fetchCuratedGameFavoritesFromRepo,
  persistCuratedGameFavoritesToRepo,
} from "../gameSearch/curatedGameFavorites";

const MAX_PLAYED = 40;
const MAX_FAVORITES = 80;
const MAX_BLACKLIST = 200;

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
  blacklistedGames: {
    key: string;
    value: OnlineGame;
  };
  gamesSession: {
    key: string;
    value: GamesSessionRecord;
  };
}

let dbPromise: Promise<IDBPDatabase<GroveeExperienceDB>> | null = null;
let repoFavoritesMerged = false;

async function syncFavoritesToRepo(): Promise<void> {
  const games = await getFavoriteGames();
  void persistCuratedGameFavoritesToRepo(games);
}

/** Merge git-backed favorites into IndexedDB (once per session). */
export async function ensureRepoGameFavoritesMerged(): Promise<void> {
  if (repoFavoritesMerged) return;
  const curated = await fetchCuratedGameFavoritesFromRepo();
  if (curated?.games.length) {
    const db = await getDb();
    for (const game of curated.games) {
      const existing = await db.get("favoriteGames", game.id);
      if (!existing) await db.put("favoriteGames", game);
    }
  }
  repoFavoritesMerged = true;
}

function getDb() {
  if (!dbPromise) {
    dbPromise = openDB<GroveeExperienceDB>("grovee-experience", 2, {
      upgrade(db, oldVersion) {
        if (oldVersion < 1) {
          db.createObjectStore("playedGames", { keyPath: "id" });
          db.createObjectStore("favoriteGames", { keyPath: "id" });
          db.createObjectStore("gamesSession", { keyPath: "key" });
        }
        if (oldVersion < 2 && !db.objectStoreNames.contains("blacklistedGames")) {
          db.createObjectStore("blacklistedGames", { keyPath: "id" });
        }
      },
    });
  }
  return dbPromise;
}

export function filterBlacklistedGames(games: OnlineGame[], blacklist: ReadonlySet<string>): OnlineGame[] {
  if (!blacklist.size) return games;
  return games.filter((g) => !blacklist.has(g.id));
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

export async function getFavoriteGames(limit?: number): Promise<OnlineGame[]> {
  await ensureRepoGameFavoritesMerged();
  const db = await getDb();
  const all = await db.getAll("favoriteGames");
  const sorted = all.sort((a, b) => a.title.localeCompare(b.title, "he"));
  return limit ? sorted.slice(0, limit) : sorted;
}

export async function isFavoriteGame(id: string): Promise<boolean> {
  const db = await getDb();
  return !!(await db.get("favoriteGames", id));
}

export async function getFavoriteIds(): Promise<Set<string>> {
  await ensureRepoGameFavoritesMerged();
  const db = await getDb();
  const all = await db.getAllKeys("favoriteGames");
  return new Set(all);
}

export async function getBlacklistedGames(): Promise<OnlineGame[]> {
  const db = await getDb();
  const all = await db.getAll("blacklistedGames");
  return all.sort((a, b) => a.title.localeCompare(b.title, "he"));
}

export async function getBlacklistedIds(): Promise<Set<string>> {
  const db = await getDb();
  const all = await db.getAllKeys("blacklistedGames");
  return new Set(all);
}

export async function isBlacklistedGame(id: string): Promise<boolean> {
  const db = await getDb();
  return !!(await db.get("blacklistedGames", id));
}

/** Returns true when the game is now blacklisted. */
export async function toggleBlacklistedGame(game: OnlineGame): Promise<boolean> {
  const db = await getDb();
  const existing = await db.get("blacklistedGames", game.id);
  if (existing) {
    await db.delete("blacklistedGames", game.id);
    return false;
  }
  await db.put("blacklistedGames", game);
  const count = await db.count("blacklistedGames");
  if (count > MAX_BLACKLIST) {
    const all = await db.getAll("blacklistedGames");
    all.sort((a, b) => a.title.localeCompare(b.title));
    await db.delete("blacklistedGames", all[0].id);
  }
  return true;
}

export async function toggleFavoriteGame(game: OnlineGame): Promise<boolean> {
  await ensureRepoGameFavoritesMerged();
  const db = await getDb();
  const existing = await db.get("favoriteGames", game.id);
  if (existing) {
    await db.delete("favoriteGames", game.id);
    void syncFavoritesToRepo();
    return false;
  }
  await db.put("favoriteGames", game);
  const count = await db.count("favoriteGames");
  if (count > MAX_FAVORITES) {
    const all = await db.getAll("favoriteGames");
    all.sort((a, b) => a.title.localeCompare(b.title));
    await db.delete("favoriteGames", all[0].id);
  }
  void syncFavoritesToRepo();
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
