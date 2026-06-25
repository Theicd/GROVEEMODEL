import type { OnlineGame } from "./types";

export const CURATED_GAME_FAVORITES_PUBLIC_PATH = "games/curatedFavorites.json";
export const CURATED_GAME_FAVORITES_API_PATH = "/api/games/curated-favorites";

export type CuratedGameFavoritesFile = {
  version: 1;
  description?: string;
  updatedAt: number;
  games: OnlineGame[];
};

function curatedGameFavoritesUrl(): string {
  const base = import.meta.env.BASE_URL || "./";
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return `${prefix}${CURATED_GAME_FAVORITES_PUBLIC_PATH}`;
}

export function emptyCuratedGameFavoritesFile(): CuratedGameFavoritesFile {
  return {
    version: 1,
    description:
      "Curated game favorites for hero rotation — source of truth in git. Auto-updated in dev when starring ☆.",
    updatedAt: 0,
    games: [],
  };
}

export async function fetchCuratedGameFavoritesFromRepo(): Promise<CuratedGameFavoritesFile | null> {
  try {
    const res = await fetch(curatedGameFavoritesUrl(), { cache: "no-store" });
    if (!res.ok) return null;
    const parsed = (await res.json()) as CuratedGameFavoritesFile;
    if (parsed.version !== 1 || !Array.isArray(parsed.games)) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function buildCuratedGameFavoritesFile(games: OnlineGame[]): CuratedGameFavoritesFile {
  const sorted = [...games].sort((a, b) => a.title.localeCompare(b.title, "he"));
  return {
    version: 1,
    description:
      "Curated game favorites for hero rotation — source of truth in git. Auto-updated in dev when starring ☆.",
    updatedAt: Date.now(),
    games: sorted,
  };
}

export async function persistCuratedGameFavoritesToRepo(
  games: OnlineGame[],
): Promise<{ ok: boolean; skipped?: boolean; error?: string }> {
  if (!import.meta.env.DEV) return { ok: true, skipped: true };

  const body = buildCuratedGameFavoritesFile(games);
  try {
    const res = await fetch(CURATED_GAME_FAVORITES_API_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      return { ok: false, error: text || `HTTP ${res.status}` };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}
