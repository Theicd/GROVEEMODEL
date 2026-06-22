import { describe, expect, it } from "vitest";
import { geocodePlace } from "./geoResolve";
import { wantsNewsHeadlineBulletsInChat } from "./openWebTopics";
import { USER_PRESENTATION_QUERIES } from "../userPresentationQueries";

describe("geoResolve Israel", () => {
  it("geocodePlace resolves ישראל to IL without homonym", async () => {
    const place = await geocodePlace("ישראל");
    expect(place?.country_code).toBe("IL");
    expect(place?.name).toBe("Israel");
  });
});

describe("wantsNewsHeadlineBulletsInChat", () => {
  it("B01 world headlines route to chat bullets", () => {
    const b01 = USER_PRESENTATION_QUERIES.find((q) => q.id === "B01")!.prompt;
    expect(wantsNewsHeadlineBulletsInChat(b01)).toBe(true);
  });
});
