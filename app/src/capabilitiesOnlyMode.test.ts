import { describe, expect, it } from "vitest";
import {
  buildCapabilitiesOnlyFallbackMessage,
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

  it("picks image rack model for capabilities default", () => {
    expect(pickCapabilitiesDefaultRackId([imageEntry])).toBe("pollinations-flux");
    expect(pickCapabilitiesDefaultRackId([])).toBeNull();
  });
});
