import { describe, expect, it } from "vitest";
import type { SearchSourceResult } from "../webSearch/types";
import {
  buildGlobeCommandFromSearch,
  parseCoordsFromNominatimText,
  shouldOpenGlobeForStructuredGeo,
} from "./searchGlobeBridge";

const placeSource = (text: string, geo?: SearchSourceResult["geo"]): SearchSourceResult => ({
  provider: "nominatim-places",
  label: "OSM",
  ok: true,
  text,
  latencyMs: 1,
  geo,
});

describe("searchGlobeBridge", () => {
  it("parses coords from nominatim text", () => {
    const geo = parseCoordsFromNominatimText(
      "1. Flughafen BER\n   52.36399, 13.50833",
    );
    expect(geo?.lat).toBeCloseTo(52.364, 2);
    expect(geo?.lon).toBeCloseTo(13.508, 2);
  });

  it("opens globe for places with coords", () => {
    const sources = [
      placeSource("1. BER station\n52.36, 13.50", { lat: 52.36, lon: 13.5, label: "BER" }),
    ];
    expect(
      shouldOpenGlobeForStructuredGeo("תחנת רכבת ליד BER", ["places"], sources),
    ).toBe(true);
  });

  it("builds flyTo from nominatim geo", () => {
    const cmd = buildGlobeCommandFromSearch(
      "תחנת רכבet BER",
      ["places"],
      [placeSource("", { lat: 52.36, lon: 13.5, label: "BER Bahnhof" })],
    );
    expect(cmd?.type).toBe("flyTo");
    if (cmd?.type === "flyTo") {
      expect(cmd.lat).toBeCloseTo(52.36);
      expect(cmd.label).toBe("BER Bahnhof");
    }
  });

  it("builds drawRoute from osrm geo", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "osrm-distance",
        label: "OSRM",
        ok: true,
        text: "מ: A\nאל: B",
        latencyMs: 1,
        geo: {
          route: [
            { lat: 52.36, lon: 13.5 },
            { lat: 52.37, lon: 13.51 },
          ],
          label: "A → B",
        },
      },
    ];
    const cmd = buildGlobeCommandFromSearch(
      "איך מגיעים משדה התעופה לתחנת הרכבet",
      ["distance"],
      sources,
    );
    expect(cmd?.type).toBe("drawRoute");
  });
});
