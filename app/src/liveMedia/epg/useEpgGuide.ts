import { useEffect, useState } from "react";
import type { UnifiedSearchHit } from "../../searchResults/types";
import { countGuideEntriesWithData, loadEpgGuide, type EpgGuideEntry } from "./epgGuideStore";
import { warmMjhEpgCaches } from "./epgService";

export function useEpgGuide(hits: UnifiedSearchHit[], enabled: boolean) {
  const [entries, setEntries] = useState<EpgGuideEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [readyCount, setReadyCount] = useState(0);
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const hitKey = hits.map((h) => `${h.id}:${h.title}:${h.mediaPlayUrl || h.url || ""}`).join("|");

  useEffect(() => {
    if (!enabled || !hits.length) {
      setProgress({ loaded: 0, total: 0 });
      return;
    }
    void warmMjhEpgCaches();
  }, [enabled, hitKey, hits.length]);

  useEffect(() => {
    if (!enabled || !hits.length) return;
    let alive = true;
    setLoading(true);
    void loadEpgGuide(hits, (partial, loaded, total) => {
      if (!alive) return;
      setEntries(partial);
      setReadyCount(countGuideEntriesWithData(partial));
      setProgress({ loaded, total });
    }).then((loaded) => {
      if (!alive) return;
      setEntries(loaded);
      setReadyCount(countGuideEntriesWithData(loaded));
      setProgress({ loaded: loaded.length, total: loaded.length });
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, [enabled, hitKey, hits]);

  return { entries, loading, readyCount, progress, hasData: readyCount > 0 };
}
