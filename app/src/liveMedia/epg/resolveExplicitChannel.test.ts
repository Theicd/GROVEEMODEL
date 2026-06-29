import { describe, expect, it } from "vitest";
import { resolveExplicitChannel } from "./epgService";
import type { EpgChannelRef } from "./types";

const src = "mjh-samsung-gb";

describe("resolveExplicitChannel (self-healing ids)", () => {
  it("uses channelId when its name still matches the canonical name", () => {
    const channels: EpgChannelRef[] = [
      { id: "GBBA33000557H", name: "Moviesphere by Lionsgate", sourceKey: src },
      { id: "GBBC2300003ZQ", name: "Entertainment Hub", sourceKey: src },
    ];
    const ch = resolveExplicitChannel(channels, {
      sourceKey: src,
      channelId: "GBBA33000557H",
      channelName: "Moviesphere by Lionsgate",
    });
    expect(ch?.id).toBe("GBBA33000557H");
  });

  it("re-resolves by name when the hardcoded id was recycled to another channel", () => {
    // Old id GBBC2300003ZQ now belongs to "Entertainment Hub"; MovieSphere moved to a new id.
    const channels: EpgChannelRef[] = [
      { id: "GBBC2300003ZQ", name: "Entertainment Hub", sourceKey: src },
      { id: "GB_NEW_999", name: "Moviesphere by Lionsgate", sourceKey: src },
    ];
    const ch = resolveExplicitChannel(channels, {
      sourceKey: src,
      channelId: "GBBC2300003ZQ",
      channelName: "Moviesphere by Lionsgate",
    });
    expect(ch?.id).toBe("GB_NEW_999");
    expect(ch?.name).toMatch(/moviesphere/i);
  });

  it("does not match an unrelated channel when name is absent and id is gone", () => {
    const channels: EpgChannelRef[] = [{ id: "OTHER", name: "Americas Got Talent", sourceKey: src }];
    const ch = resolveExplicitChannel(channels, { sourceKey: src, channelId: "MISSING" });
    expect(ch).toBeNull();
  });
});
