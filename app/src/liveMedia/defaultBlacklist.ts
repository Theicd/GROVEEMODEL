import { channelPassesHeEnCatalog, radioPassesHeEnCatalog } from "./heEnCatalogFilter";
import type { Channel, RadioStation } from "./types";

export function matchesDefaultBlacklistChannel(c: Channel): boolean {
  return !channelPassesHeEnCatalog(c);
}

export function matchesDefaultBlacklistRadio(r: RadioStation): boolean {
  return !radioPassesHeEnCatalog(r);
}

export function collectDefaultBlacklistIds(channels: Channel[], radio: RadioStation[]): {
  channelIds: string[];
  radioIds: string[];
} {
  return {
    channelIds: channels.filter(matchesDefaultBlacklistChannel).map((c) => c.id),
    radioIds: radio.filter(matchesDefaultBlacklistRadio).map((r) => r.id),
  };
}
