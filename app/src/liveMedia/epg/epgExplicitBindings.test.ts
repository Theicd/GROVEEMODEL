import { describe, expect, it } from "vitest";
import { explicitEpgTargets } from "./epgExplicitBindings";

describe("epgExplicitBindings", () => {
  it("binds ion mystery stream to verified XMLTV ids", () => {
    const stream =
      "https://cdn-uw2-prod.tsv2.amagi.tv/linear/amg01438-ewscrippscompan-ionmystery-tablo/playlist.m3u8";
    const targets = explicitEpgTargets("WFXT662.us@SD", stream);
    expect(targets.some((t) => t.sourceKey === "mjh-plex-us" && t.channelId.includes("62b45f15"))).toBe(true);
  });

  it("binds FIFA+ tvg-id to plex/roku feeds", () => {
    const targets = explicitEpgTargets("FIFAPlus.uk@UnitedStates", "");
    expect(targets.some((t) => t.sourceKey === "mjh-roku")).toBe(true);
    expect(targets.some((t) => t.sourceKey === "mjh-plex-us")).toBe(true);
  });

  it("binds Saved by the Bell xumo stream to roku XMLTV id", () => {
    const stream = "https://xumo-xumoent-vc-111-0pd1g.fast.nbcuni.com/live/master.m3u8";
    const targets = explicitEpgTargets(undefined, stream);
    expect(targets.some((t) => t.sourceKey === "mjh-roku" && t.channelId === "05a58f8f0d1b55999a9ab0e9caae8a47")).toBe(
      true,
    );
  });
});
