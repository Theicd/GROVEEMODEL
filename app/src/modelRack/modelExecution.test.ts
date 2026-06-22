import { describe, expect, it, vi } from "vitest";
import { buildPollinationsUrl } from "../cloudImage";
import { executeRackModel } from "./modelExecution";
import type { RackModelEntry } from "./modelRack";

vi.mock("../localImageGen", () => ({
  generateSdTurboPng: vi.fn(async () => ({ ok: true, objectUrl: "blob:mock" })),
}));

describe("modelExecution", () => {
  it("generates pollinations markdown for flux model", async () => {
    const model: RackModelEntry = {
      id: "pollinations-flux",
      label: "FLUX",
      modality: "image",
      adapter: "pollinations",
      status: "ready",
      source: "builtin",
      pollinationsModel: "flux",
      addedAt: 0,
    };
    const out = await executeRackModel(model, "a red cat on mars", () => {});
    expect(out.ok).toBe(true);
    if (out.ok) {
      expect(out.content).toContain("![Generated](");
      expect(out.content).toContain(buildPollinationsUrl({ prompt: "a red cat on mars", model: "flux" }));
    }
  });

  it("rejects empty prompt", async () => {
    const model: RackModelEntry = {
      id: "pollinations-flux",
      label: "FLUX",
      modality: "image",
      adapter: "pollinations",
      status: "ready",
      source: "builtin",
      pollinationsModel: "flux",
      addedAt: 0,
    };
    const out = await executeRackModel(model, "   ", () => {});
    expect(out.ok).toBe(false);
  });

});
