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

describe("geoResolve Brazil", () => {
  it("geocodePlace resolves ברזיל to Brazil country not Indiana US", async () => {
    const place = await geocodePlace("ברזיל");
    expect(place?.country_code).toBe("BR");
    expect(place?.name).toBe("Brazil");
  });

  it("geocodePlace resolves Hebrew Rio to Rio de Janeiro", async () => {
    const place = await geocodePlace("ריו דה ז׳ניירו");
    expect(place?.country_code).toBe("BR");
    expect(place?.name).toBe("Rio de Janeiro");
  });

  it("geocodePlace resolves Hebrew São Paulo", async () => {
    const place = await geocodePlace("סאו פאולו");
    expect(place?.country_code).toBe("BR");
    expect(place?.name).toBe("São Paulo");
  });
});

describe("wantsNewsHeadlineBulletsInChat", () => {
  it("B01 world headlines route to chat bullets", () => {
    const b01 = USER_PRESENTATION_QUERIES.find((q) => q.id === "B01")!.prompt;
    expect(wantsNewsHeadlineBulletsInChat(b01)).toBe(true);
  });
});
