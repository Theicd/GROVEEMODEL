/** @vitest-environment jsdom */
import { describe, expect, it } from "vitest";
import { channelMayHaveEpg, resolveEpgMatchTitles, resolveIptvOrgChannelId } from "./channelAliases";
import { normalizeChannelTitle, normalizeForMatch, stripTvgFeed } from "./normalize";
import { matchChannelInXml, parseXmltvPrograms } from "./xmltvParse";

const SAMPLE_XML = `<?xml version="1.0" encoding="UTF-8"?>
<tv>
  <channel id="mjh-10-cops"><display-name>COPS</display-name></channel>
  <programme channel="mjh-10-cops" start="20260625120000 +0000" stop="20260625130000 +0000">
    <title>On Patrol</title>
    <desc>Officers respond to a call.</desc>
  </programme>
</tv>`;

describe("epg normalize", () => {
  it("strips quality suffixes and broken user-agent noise", () => {
    expect(normalizeChannelTitle("Now 14 (1080p)")).toBe("Now 14");
    expect(normalizeChannelTitle('like Gecko) Chrome/145.0.0.0",24 Hour Free Movies (720p)')).toContain(
      "24 Hour Free Movies",
    );
  });

  it("strips @feed from tvg ids", () => {
    expect(stripTvgFeed("Channel9.il@SD")).toBe("Channel9.il");
  });
});

describe("channelAliases", () => {
  it("maps favorite names to iptv-org ids", () => {
    expect(resolveIptvOrgChannelId("COPS")).toBe("Cops.us");
    expect(resolveIptvOrgChannelId("The Pet Collective")).toBe("ThePetCollective.us");
    expect(resolveIptvOrgChannelId("Reshet 13 (720p)")).toBe("Channel13.il");
  });

  it("hints EPG for known US channels before probe", () => {
    expect(channelMayHaveEpg("COPS")).toBe(true);
    expect(channelMayHaveEpg("The Pet Collective")).toBe(true);
    expect(channelMayHaveEpg("Reshet 13 (720p)")).toBe(false);
  });

  it("prefers explicit tvg-id", () => {
    expect(resolveIptvOrgChannelId("Foo", "Kan11.il@SD")).toBe("Kan11.il");
  });

  it("resolves WFXT-DT2 / ION Mystery EPG match titles", () => {
    const stream =
      "https://cdn-uw2-prod.tsv2.amagi.tv/linear/amg01438-ewscrippscompan-ionmystery-tablo/playlist.m3u8";
    const titles = resolveEpgMatchTitles("WFXT-DT2 (1080p)", "WFXT662.us@SD", stream);
    expect(titles).toContain("ION Mystery");
    expect(channelMayHaveEpg("WFXT-DT2 (1080p)", "WFXT662.us@SD", stream)).toBe(true);
  });

  it("resolves FIFA+ and Entertainment Tonight hints", () => {
    expect(resolveEpgMatchTitles("FIFA+ United States (720p)", "FIFAPlus.uk@UnitedStates", "")).toContain(
      "FIFA+",
    );
    expect(
      resolveEpgMatchTitles("Entertainment Tonight (1080p)", "EntertainmentTonight.us@SD", ""),
    ).toContain("ET");
    expect(
      resolveEpgMatchTitles("AMC Absolute Reality", "AbsoluteRealitybyWETV.us@SD", "https://wetv.example/playlist.m3u8"),
    ).toContain("All Reality We TV");
  });
});

describe("xmltvParse", () => {
  it("matches channel by display name", () => {
    const ch = matchChannelInXml(SAMPLE_XML, "mjh-all", "COPS", normalizeForMatch("COPS"));
    expect(ch?.id).toBe("mjh-10-cops");
  });

  it("does not match short substrings inside longer titles (Reshet 13 vs ET)", () => {
    const xml = `<?xml version="1.0"?><tv>
      <channel id="pluto-et"><display-name>ET</display-name></channel>
    </tv>`;
    const ch = matchChannelInXml(xml, "mjh-pluto-us", "Reshet 13 (720p)", "reshet 13");
    expect(ch).toBeNull();
  });

  it("parses programmes for channel id", () => {
    const programs = parseXmltvPrograms(SAMPLE_XML, "mjh-10-cops");
    expect(programs).toHaveLength(1);
    expect(programs[0].title).toBe("On Patrol");
  });

  it("decodes xml entities in programme titles", () => {
    const xml = `<?xml version="1.0"?><tv>
      <programme start="20260625120000 +0000" stop="20260625130000 +0000" channel="x1">
        <title>Stan &amp; Ollie</title>
      </programme>
    </tv>`;
    const programs = parseXmltvPrograms(xml, "x1");
    expect(programs[0].title).toBe("Stan & Ollie");
  });

  it("parses season, episode, and description", () => {
    const xml = `<?xml version="1.0"?><tv>
      <programme start="20260625120000 +0000" stop="20260625130000 +0000" channel="x1">
        <title>Law &amp; Order</title>
        <sub-title>The Witness</sub-title>
        <desc>A witness comes forward.</desc>
        <episode-num system="onscreen">S12E05</episode-num>
      </programme>
    </tv>`;
    const programs = parseXmltvPrograms(xml, "x1");
    expect(programs[0].season).toBe(12);
    expect(programs[0].episode).toBe(5);
    expect(programs[0].description).toBe("A witness comes forward.");
    expect(programs[0].subTitle).toBe("The Witness");
  });
});
