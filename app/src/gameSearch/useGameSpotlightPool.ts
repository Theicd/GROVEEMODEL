import { useCallback, useEffect, useState } from "react";
import type { OnlineGame } from "./types";
import { loadFeaturedFallback, randomOnlineGames } from "./archiveBrowser";

const ROTATE_MS = 8000;
const REFRESH_POOL_MS = 5 * 60 * 1000;

export function useGameSpotlightPool() {
  const [games, setGames] = useState<OnlineGame[]>([]);
  const [index, setIndex] = useState(0);

  const loadPool = useCallback(async () => {
    try {
      const result = await randomOnlineGames(8, "featured");
      if (result.games.length) {
        setGames(result.games);
        setIndex(0);
        return;
      }
    } catch {
      /* fallback below */
    }
    const fb = await loadFeaturedFallback();
    if (fb.length) {
      setGames(fb.slice(0, 8));
      setIndex(0);
    }
  }, []);

  useEffect(() => {
    void loadPool();
  }, [loadPool]);

  useEffect(() => {
    const refreshId = window.setInterval(() => {
      void loadPool();
    }, REFRESH_POOL_MS);
    return () => window.clearInterval(refreshId);
  }, [loadPool]);

  useEffect(() => {
    if (games.length < 2) return;
    const id = window.setInterval(() => {
      setIndex((i) => (i + 1) % games.length);
    }, ROTATE_MS);
    return () => window.clearInterval(id);
  }, [games.length]);

  const current = games[index] ?? games[0] ?? null;

  return { games, index, setIndex, current, loadPool };
}
