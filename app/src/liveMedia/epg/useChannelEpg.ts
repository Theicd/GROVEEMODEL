import { useEffect, useState } from "react";
import type { UnifiedSearchHit } from "../../searchResults/types";
import { channelHasEpg, channelMayHaveEpg, hitToEpgLookup, warmMjhEpgCaches } from "./epgService";

export function useChannelEpgAvailable(hit: UnifiedSearchHit | null, enabled: boolean): boolean {
  const lookup = hit ? hitToEpgLookup(hit) : null;
  const [available, setAvailable] = useState(() =>
    lookup ? channelMayHaveEpg(lookup.title, lookup.tvgId, lookup.streamUrl) : false,
  );

  useEffect(() => {
    if (!enabled || !hit) {
      setAvailable(false);
      return;
    }
    const nextLookup = hitToEpgLookup(hit);
    if (!nextLookup) {
      setAvailable(false);
      return;
    }

    const optimistic = channelMayHaveEpg(nextLookup.title, nextLookup.tvgId, nextLookup.streamUrl);
    setAvailable(optimistic);
    void warmMjhEpgCaches(nextLookup.streamUrl);

    let alive = true;
    void channelHasEpg(nextLookup).then((ok) => {
      if (alive) setAvailable(ok || optimistic);
    });
    return () => {
      alive = false;
    };
  }, [enabled, hit?.id, hit?.title, hit?.mediaPlayUrl, hit?.url, hit?.meta?.tvgId]);

  return available;
}
