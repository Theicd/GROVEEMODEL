import { describe, expect, it } from "vitest";
import { preferredOffsetHoursFromTvgFeed, streamEpgAffinityBonus } from "./epgStreamAffinity";

describe("epgStreamAffinity", () => {
  it("boosts pluto stream matching pluto channel", () => {
    expect(
      streamEpgAffinityBonus("https://service.pluto.tv/master.m3u8", "Comedy Central Pluto TV", "mjh-pluto-us"),
    ).toBeGreaterThan(0);
  });

  it("returns eastern offset hints for @East tvg-id", () => {
    expect(preferredOffsetHoursFromTvgFeed("ComedyCentral.us@East")).toContain(4);
  });
});
