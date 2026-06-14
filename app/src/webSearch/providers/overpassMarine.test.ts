import { describe, expect, it, vi } from "vitest";
import { buildPersistedAssistantPayload } from "../../artifacts";
import { GROVEE_CHAT_SYSTEM, LANGUAGE_RULE_MARKER } from "../../characterPrompts";

describe("artifacts prompt leak", () => {
  it("blocks HTML artifact when content is leaked system prompt", () => {
    const leaked = `\`\`\`html\nsystem\n\n${GROVEE_CHAT_SYSTEM.slice(0, 200)}\n\`\`\``;
    const payload = buildPersistedAssistantPayload(leaked, false);
    expect(payload.artifact).toBeNull();
    expect(payload.content).not.toContain(LANGUAGE_RULE_MARKER);
  });
});

describe("fetchOverpassMarineSearch", () => {
  it("formats harbour and buoy counts for regional query", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            elements: [
              { type: "node", id: 1, lat: 32.82, lon: 35.0, tags: { "seamark:type": "harbour", name: "Haifa" } },
              { type: "node", id: 2, lat: 32.81, lon: 35.01, tags: { "seamark:type": "buoy_lateral", name: "Buoy A" } },
              { type: "node", id: 3, lat: 32.83, lon: 35.02, tags: { "seamark:type": "light_major", name: "Light" } },
            ],
          }),
          { status: 200 },
        ),
      ),
    );
    const { fetchOverpassMarineSearch } = await import("./overpassMarine");
    const result = await fetchOverpassMarineSearch("כמה מצופים במפרץ חיפה?");
    expect(result.ok).toBe(true);
    expect(result.provider).toBe("osm-overpass-marine");
    expect(result.text).toMatch(/תשתיות ימיות/);
    vi.unstubAllGlobals();
  });
});
