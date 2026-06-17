import { describe, expect, it } from "vitest";

import { GROVEE_INFO_CARDS } from "./groveeInfoContent";

describe("groveeInfoContent", () => {
  it("has distinct card ids and required fields", () => {
    const ids = GROVEE_INFO_CARDS.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const card of GROVEE_INFO_CARDS) {
      expect(card.title.length).toBeGreaterThan(2);
      expect(card.body.length).toBeGreaterThan(10);
    }
  });
});
