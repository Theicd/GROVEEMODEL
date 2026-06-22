import { useEffect, useState } from "react";
import { PLUGIN_STATUS_EVENT } from "./events";
import { getPluginHealthSnapshot, pollAllPluginsHealth } from "./healthCoordinator";
import type { PluginHealthSnapshot } from "./types";

/** Poll local companion plugins (health + auto-detect). */
export function usePluginHealthPoll(enabled = true): PluginHealthSnapshot {
  const [snapshot, setSnapshot] = useState<PluginHealthSnapshot>(() => getPluginHealthSnapshot());

  useEffect(() => {
    if (!enabled || typeof window === "undefined") return;

    const refresh = () => setSnapshot(getPluginHealthSnapshot());
    const tick = () => void pollAllPluginsHealth().then(setSnapshot);

    tick();
    const intervalId = window.setInterval(tick, 10_000);
    const onVisible = () => {
      if (document.visibilityState === "visible") tick();
    };
    window.addEventListener(PLUGIN_STATUS_EVENT, refresh);
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      window.clearInterval(intervalId);
      window.removeEventListener(PLUGIN_STATUS_EVENT, refresh);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [enabled]);

  return snapshot;
};
