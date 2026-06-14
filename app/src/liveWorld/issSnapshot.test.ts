import { describe, expect, it } from "vitest";
import { clearLiveWorldSnapshotCache, setLiveWorldSnapshot } from "./snapshotStore";
import { issSearchResultFromLiveWorld } from "./issSnapshot";
import { buildCapabilityLiveReply } from "../webSearch/capabilityReplyMessages";

describe("issSnapshot", () => {
  it("returns ISS from live world cache before API", () => {
    clearLiveWorldSnapshotCache();
    setLiveWorldSnapshot({
      fetchedAt: Date.now() - 60_000,
      source: "globe",
      iss: { lat: -12.34, lon: 56.78, altitudeKm: 419, velocityKmh: 27600 },
    });

    const result = issSearchResultFromLiveWorld("היכן נמצאת תחנת החלל הבינלאומית כרגע?");
    expect(result?.ok).toBe(true);
    expect(result?.text).toContain("-12.34");
    expect(result?.text).toContain("ANSWER (ISS position)");
    expect(result?.provider).toBe("iss-tracker");
  });

  it("builds canned ISS reply from cache when provider failed", () => {
    clearLiveWorldSnapshotCache();
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "globe",
      iss: { lat: 1.23, lon: 4.56, altitudeKm: 420 },
    });

    const reply = buildCapabilityLiveReply(
      "היכן נמצאת תחנת החלל הבינלאומית כרגע?",
      ["satellite"],
      [{ provider: "iss-tracker", label: "x", ok: false, text: "", error: "timeout", latencyMs: 7 }],
    );
    expect(reply).toMatch(/1\.23|4\.56/);
    expect(reply).not.toMatch(/לא הצלחתי לטעון/);
  });
});
