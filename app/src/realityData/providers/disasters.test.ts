import { describe, expect, it } from "vitest";
import { isGdacsEventLive, parseGdacsFeatures } from "./disasters";
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

  it("parseGdacsFeatures extracts GDACS ids and geometry url", () => {
    const items = parseGdacsFeatures([
      {
        properties: {
          eventname: "BAVI-26",
          alertlevel: "Red",
          eventtype: "TC",
          eventid: 1001279,
          episodeid: 8,
          url: {
            geometry:
              "https://www.gdacs.org/gdacsapi/api/polygons/getgeometry?eventtype=TC&eventid=1001279&episodeid=8",
            report: "https://www.gdacs.org/report.aspx?eventid=1001279",
          },
          severitydata: { severitytext: "Hurricane (maximum wind speed of 185 km/h)" },
        },
      },
    ]);
    expect(items[0].eventId).toBe(1001279);
    expect(items[0].episodeId).toBe(8);
    expect(items[0].geometryUrl).toContain("getgeometry");
    expect(items[0].severityText).toContain("Hurricane");
  });

  it("isGdacsEventLive keeps current cyclones and drops stale EQ", () => {
    const now = Date.now();
    const storm: ReturnType<typeof parseGdacsFeatures>[0] = {
      eventName: "BAVI-26",
      country: "Guam",
      alertLevel: "Red",
      eventType: "TC",
      isCurrent: true,
      endTime: now + 3600000,
      dateModified: now - 60000,
    };
    const oldEq: ReturnType<typeof parseGdacsFeatures>[0] = {
      eventName: "EQ Venezuela",
      country: "Venezuela",
      alertLevel: "Red",
      eventType: "EQ",
      isCurrent: false,
      dateModified: now - 5 * 86400000,
    };
    expect(isGdacsEventLive(storm, now)).toBe(true);
    expect(isGdacsEventLive(oldEq, now)).toBe(false);
  });

  it("isGdacsEventLive drops ended episodes", () => {
    const now = Date.now();
    const ended = {
      eventName: "STORM",
      country: "X",
      alertLevel: "Orange",
      eventType: "TC",
      isCurrent: true,
      endTime: now - 1000,
    };
    expect(isGdacsEventLive(ended, now)).toBe(false);
  });
});
