/** @vitest-environment jsdom */
import { describe, expect, it } from "vitest";
import { fetchEpgSchedule, channelHasEpg, resetEpgAvailabilityCacheForTests } from "./epgService";
import { resetMjhEpgCacheForTests, warmMjhEpgCaches } from "./mjhSources";
import { resetEpgGuideIndexForTests } from "./epgGuideIndex";

describe("epgService integration", () => {
  it(
    "loads programme data for COPS from MJH",
    async () => {
      resetMjhEpgCacheForTests();
      resetEpgAvailabilityCacheForTests();
      resetEpgGuideIndexForTests();
      const input = {
        title: "COPS",
        streamUrl:
          "https://a7d6af1c184e465db4f39316a5181c1f.mediatailor.us-east-1.amazonaws.com/v1/master/0fb304b2320b25f067414d481a779b77db81760d/RakutenTV-eu_COPS/playlist.m3u8",
      };
      const schedule = await fetchEpgSchedule(input);
      expect(schedule).not.toBeNull();
      expect(schedule!.programs.length).toBeGreaterThan(0);
      expect(schedule!.programs[0].title.length).toBeGreaterThan(0);
    },
    60_000,
  );

  it(
    "hasEpg is false when only iptv-org index matches without programmes (Reshet 13)",
    async () => {
      resetMjhEpgCacheForTests();
      resetEpgAvailabilityCacheForTests();
      resetEpgGuideIndexForTests();
      const input = { title: "Reshet 13 (720p)", streamUrl: "https://example.com/stream.m3u8" };
      const has = await channelHasEpg(input);
      expect(has).toBe(false);
    },
    5_000,
  );
});
