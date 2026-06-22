// @vitest-environment node
import { describe, expect, it } from "vitest";
import { collectAisStreamShips } from "../../../vite-plugins/aisStreamProxy";

const KEY = process.env.AISSTREAM_API_KEY?.trim();

describe.skipIf(!KEY)("aisStream live (AISSTREAM_API_KEY)", () => {
  it(
    "collects ships near Haifa bay",
    async () => {
      const { ships, error } = await collectAisStreamShips(
        KEY!,
        32.72,
        32.92,
        34.92,
        35.12,
        12_000,
      );
      expect(error).toBeUndefined();
      expect(ships.length).toBeGreaterThan(0);
      expect(ships[0].source).toBe("aisstream");
      expect(Number.isFinite(ships[0].lat)).toBe(true);
    },
    20_000,
  );
});
