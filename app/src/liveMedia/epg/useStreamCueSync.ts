import { useEffect, useState } from "react";
import { fetchHlsCueState, type HlsCueState } from "./hlsCueSync";

const POLL_MS = 8_000;

export function useStreamCueSync(streamUrl: string | undefined, enabled: boolean): HlsCueState | null {
  const [cue, setCue] = useState<HlsCueState | null>(null);

  useEffect(() => {
    if (!enabled || !streamUrl || !/\.m3u8(\?|$)/i.test(streamUrl)) {
      setCue(null);
      return;
    }

    let alive = true;
    const poll = async () => {
      const state = await fetchHlsCueState(streamUrl);
      if (alive) setCue(state);
    };

    void poll();
    const id = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [streamUrl, enabled]);

  return cue;
}
