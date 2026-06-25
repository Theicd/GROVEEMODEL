import { describe, expect, it } from "vitest";
import { buildRegionalRadioLineup, radioMatchesRegion } from "./cableTunerRadio";
import type { RadioStation } from "./types";

function station(partial: Partial<RadioStation> & Pick<RadioStation, "id" | "name">): RadioStation {
  return {
    favicon: "",
    tags: [],
    country: "",
    countrycode: "",
    language: "",
    stream: "https://example.com/stream",
    type: "radio",
    favorite: false,
    ...partial,
  };
}

describe("cableTunerRadio", () => {
  it("prefers regional stations for Israel", () => {
    const list = [
      station({ id: "de1", name: "Berlin FM", countrycode: "de", country: "Germany" }),
      station({ id: "il1", name: "Galgalatz", countrycode: "il", country: "Israel", votes: 900 }),
      station({ id: "il2", name: "Kol Israel", countrycode: "il", country: "Israel", votes: 800 }),
    ];
    const hits = buildRegionalRadioLineup(list, "il", 10);
    expect(hits[0]?.title).toMatch(/Galgalatz|Kol Israel/);
    expect(hits.some((h) => h.title === "Berlin FM")).toBe(false);
  });

  it("includes favorites from any region first", () => {
    const list = [
      station({ id: "de1", name: "Berlin FM", countrycode: "de", favorite: true }),
      station({ id: "il1", name: "Galgalatz", countrycode: "il" }),
    ];
    const hits = buildRegionalRadioLineup(list, "il", 10);
    expect(hits[0]?.title).toBe("Berlin FM");
  });

  it("matches Germany aliases", () => {
    expect(radioMatchesRegion(station({ id: "x", name: "X", countrycode: "de" }), "de")).toBe(true);
    expect(radioMatchesRegion(station({ id: "x", name: "X", country: "Germany" }), "de")).toBe(true);
  });
});
