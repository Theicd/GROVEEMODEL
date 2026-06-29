import { describe, expect, it } from "vitest";
import { explicitEpgTargets } from "./epgExplicitBindings";

describe("epgExplicitBindings", () => {
  it("binds ion mystery stream to verified XMLTV ids", () => {
    const stream =
      "https://cdn-uw2-prod.tsv2.amagi.tv/linear/amg01438-ewscrippscompan-ionmystery-tablo/playlist.m3u8";
    const targets = explicitEpgTargets("WFXT662.us@SD", stream);
    expect(targets.some((t) => t.sourceKey === "mjh-plex-us" && t.channelId?.includes("62b45f15"))).toBe(true);
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

  it("binds History Hunters Rakuten stream to UK XMLTV id", () => {
    const stream =
      "https://amg00841-amg00841c7-rakuten-uk-2820.playouts.now.amagi.tv/playlist/amg00841-aeemeafast-historyhuntersrakuten-rakutenuk/playlist.m3u8";
    const targets = explicitEpgTargets("history-hunters", stream);
    expect(targets.some((t) => t.sourceKey === "rakuten-uk" && t.channelId === "history-hunters")).toBe(true);
  });

  it("binds MovieSphere UK to Samsung GB feed by canonical name (not US)", () => {
    const stream = "https://moviesphereuk-samsunguk.amagi.tv/playlist.m3u8";
    const targets = explicitEpgTargets("MovieSphere.us@UK", stream);
    expect(targets[0].sourceKey).toBe("mjh-samsung-gb");
    expect(targets[0].channelName).toMatch(/moviesphere/i);
    expect(targets.some((t) => t.sourceKey === "mjh-samsung-us")).toBe(true);
  });

  it("binds MovieSphere US feed to US Samsung with canonical name", () => {
    const targets = explicitEpgTargets("MovieSphere.us@US", "");
    expect(targets[0].sourceKey).toBe("mjh-samsung-us");
    expect(targets[0].channelName).toMatch(/moviesphere/i);
  });

  it("binds Comedy Central East to the linear epg.pw feed first, Pluto/Roku as fallback", () => {
    const stream = "http://23.237.104.106:8080/USA_COMEDY_CENTRAL/index.m3u8";
    const targets = explicitEpgTargets("ComedyCentral.us@East", stream);
    expect(targets[0].sourceKey).toBe("epgpw-comedycentral-east");
    expect(targets[0].feedUrl).toMatch(/epg\.pw/i);
    expect(targets[0].channelName).toMatch(/comedy central/i);
    expect(targets.some((t) => t.sourceKey === "mjh-pluto-us")).toBe(true);
    expect(targets.some((t) => t.sourceKey === "mjh-roku")).toBe(true);
  });
});
