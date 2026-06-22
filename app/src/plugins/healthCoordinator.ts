import type { GroveePlugin, PluginHealthSnapshot } from "./types";
import { searchCompanionPlugin } from "./search-companion";
import { dispatchPluginStatusEvent } from "./events";

/** All registered GROVEE plugins — add future plugins here. */
export const GROVEE_PLUGINS: GroveePlugin[] = [searchCompanionPlugin];

export const getPluginById = (id: string): GroveePlugin | undefined =>
  GROVEE_PLUGINS.find((p) => p.id === id);

const healthSnapshot: PluginHealthSnapshot = {};

export const getPluginHealthSnapshot = (): PluginHealthSnapshot => ({ ...healthSnapshot });

export const pollAllPluginsHealth = async (): Promise<PluginHealthSnapshot> => {
  for (const plugin of GROVEE_PLUGINS) {
    const result = await plugin.probeHealth();
    healthSnapshot[plugin.id] = { ...result, checkedAt: Date.now() };
    if (result.status === "online" || result.status === "degraded") {
      plugin.onDetectedOnline?.();
    }
  }
  dispatchPluginStatusEvent();
  return getPluginHealthSnapshot();
};

export const pollPluginHealth = async (id: string): Promise<PluginHealthSnapshot> => {
  const plugin = getPluginById(id);
  if (!plugin) return getPluginHealthSnapshot();
  const result = await plugin.probeHealth();
  healthSnapshot[id] = { ...result, checkedAt: Date.now() };
  if (result.status === "online" || result.status === "degraded") {
    plugin.onDetectedOnline?.();
  }
  dispatchPluginStatusEvent();
  return getPluginHealthSnapshot();
};

export const resetPluginHealthSnapshot = (): void => {
  for (const key of Object.keys(healthSnapshot)) {
    delete healthSnapshot[key];
  }
};
