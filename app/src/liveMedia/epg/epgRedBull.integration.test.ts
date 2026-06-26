/** @vitest-environment jsdom */
import { describe, expect, it } from "vitest";
import { fetchEpgSchedule, resetEpgAvailabilityCacheForTests } from "./epgService";
import { resetMjhEpgCacheForTests, warmMjhEpgCaches } from "./mjhSources";

describe("epgService Red Bull TV", () => {
  it(
    "loads programme data for Red Bull TV from MJH",
    async () => {
      resetMjhEpgCacheForTests();
      resetEpgAvailabilityCacheForTests();
      const input = {
        title: "Red Bull TV (1080p)",
        streamUrl:
          "https://3ea22335.wurl.com/master/f36d25e7e52f1ba8d7e56eb859c636563214f541/UmFrdXRlblRWLWdiX1JlZEJ1bGxUVl9ITFM/playlist.m3u8",
      };
      await warmMjhEpgCaches(input.streamUrl);
      const t0 = Date.now();
      const schedule = await fetchEpgSchedule(input);
      const ms = Date.now() - t0;
      expect(schedule).not.toBeNull();
      expect(schedule!.programs.length).toBeGreaterThan(0);
      expect(schedule!.programs[0].title.length).toBeGreaterThan(0);
      expect(ms).toBeLessThan(20_000);
    },
    120_000,
  );
});
