import { describe, expect, it } from "vitest";
import { parseGdacsFeatures } from "./disasters";

describe("GDACS disasters provider", () => {
  it("parseGdacsFeatures coerces string fields and object urls", () => {
    const items = parseGdacsFeatures([
      {
        properties: {
          eventname: "Earthquake in Japan",
          country: "Japan",
          alertlevel: "Red",
          eventtype: "EQ",
          url: { report: "https://www.gdacs.org/report.aspx?eventid=42" },
        },
      },
    ]);
    expect(items).toHaveLength(1);
    expect(items[0].eventName).toBe("Earthquake in Japan");
    expect(items[0].alertLevel).toBe("Red");
    expect(items[0].url).toBe("https://www.gdacs.org/report.aspx?eventid=42");
  });

  it("parseGdacsFeatures uses gdacs home when url missing", () => {
    const items = parseGdacsFeatures([{ properties: { eventname: "Storm", alertlevel: "Green" } }]);
    expect(items[0].url).toBe("https://www.gdacs.org");
  });
});
