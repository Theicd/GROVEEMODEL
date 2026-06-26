/** @vitest-environment jsdom */
import { describe, expect, it } from "vitest";
import { channelHasEpg, resetEpgAvailabilityCacheForTests } from "./epgService";
import { resetMjhEpgCacheForTests, warmMjhEpgCaches } from "./mjhSources";

describe("epgService probe speed", () => {
  it(
    "channelHasEpg is fast after cache warm (COPS)",
    async () => {
      resetMjhEpgCacheForTests();
      resetEpgAvailabilityCacheForTests();
      const input = {
        title: "COPS",
        streamUrl:
          "https://a7d6af1c184e465db4f39316a5181c1f.mediatailor.us-east-1.amazonaws.com/v1/master/0fb304b2320b25f067414d481a779b77db81760d/RakutenTV-eu_COPS/playlist.m3u8",
      };
      await warmMjhEpgCaches(input.streamUrl);
      const t0 = Date.now();
      const has = await channelHasEpg(input);
      const ms = Date.now() - t0;
      expect(has).toBe(true);
      expect(ms).toBeLessThan(3_000);
    },
    90_000,
  );
});
