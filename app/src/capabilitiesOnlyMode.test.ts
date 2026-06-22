import { describe, expect, it } from "vitest";
import {
  buildCapabilitiesOnlyFallbackMessage,
  buildCapabilitiesWelcomeMessage,
  pickCapabilitiesDefaultRackId,
} from "./capabilitiesOnlyMode";
import type { RackModelEntry } from "./modelRack/modelRack";

const imageEntry: RackModelEntry = {
  id: "pollinations-flux",
  label: "Flux",
  modality: "image",
  adapter: "pollinations",
  status: "ready",
  source: "cloud-scan",
  pollinationsModel: "flux",
  addedAt: 1,
};

describe("capabilitiesOnlyMode", () => {
  it("builds Hebrew fallback with failure hint", () => {
    const msg = buildCapabilitiesOnlyFallbackMessage("WebGPU failed");
    expect(msg).toContain("WebGPU failed");
    expect(msg).toContain("משחקים");
    expect(msg).toContain("תמונות");
  });

  it("builds Hebrew welcome toast ending with no-chat note", () => {
    const msg = buildCapabilitiesWelcomeMessage("WebGPU failed");
    expect(msg).toContain("ברוך הבא");
    expect(msg).toContain("משחקים");
    expect(msg).toContain("ללא מודל שיחה");
    expect(msg).toContain("WebGPU failed");
  });

  it("picks image rack model for capabilities default", () => {
    expect(pickCapabilitiesDefaultRackId([imageEntry])).toBe("pollinations-flux");
    expect(pickCapabilitiesDefaultRackId([])).toBeNull();
  });
});
