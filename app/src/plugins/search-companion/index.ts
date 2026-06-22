import type { GroveePlugin } from "../types";
import { getSearchCompanionUrl, setSearchCompanionUrl } from "./companionSettings";
import { isSearchCompanionReachable, probeSearchCompanionHealth } from "./health";
import { SEARCH_COMPANION_MANIFEST } from "./manifest";
import { probeOpenSerpSearch } from "../../webSearch/providers/openserp";

export const searchCompanionPlugin: GroveePlugin = {
  ...SEARCH_COMPANION_MANIFEST,
  probeHealth: probeSearchCompanionHealth,
  probeSearch: async (query = "webgpu browser ai") => {
    const out = await probeOpenSerpSearch(query);
    return {
      ok: out.ok,
      messageHe: out.messageHe,
      hitCount: out.hitCount,
    };
  },
  getBaseUrl: getSearchCompanionUrl,
  setBaseUrl: setSearchCompanionUrl,
  isActive: () => !!getSearchCompanionUrl() || isSearchCompanionReachable(),
  onDetectedOnline: () => {
    if (!getSearchCompanionUrl()) {
      setSearchCompanionUrl(SEARCH_COMPANION_MANIFEST.defaultBaseUrl);
    }
  },
};
