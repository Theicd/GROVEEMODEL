import { describe, expect, it } from "vitest";
import {
  GEMMA_RACK_ID,
  loadModelRack,
  mergeWithBuiltinRack,
  modalityFromPipeline,
  pickableRackModels,
  rackEntryFromHfHit,
  rackEntryTagLabel,
  rackIdFromHfModel,
  rackIdFromPollinations,
  summarizeRackCounts,
  upsertFreeRackModel,
} from "./modelRack";
import { SMOLLM_RACK_ID } from "./localTextModels";
import { pollinationsEntry } from "./pollinationsScan";

describe("modelRack", () => {
  it("includes built-in Gemma and downloadable SmolLM in picker rack", () => {
    const rack = loadModelRack();
    expect(rack.some((r) => r.id === GEMMA_RACK_ID && r.source === "builtin")).toBe(true);
    expect(rack.some((r) => r.id === SMOLLM_RACK_ID && r.adapter === "hf-local-text")).toBe(true);
    expect(rack.some((r) => r.id === "pollinations-flux" && r.source === "cloud-scan")).toBe(true);
    expect(rack.some((r) => r.id === "sd-turbo-local")).toBe(false);
  });

  it("builds stable id from hf model path", () => {
    expect(rackIdFromHfModel("Qwen/Qwen2.5-Coder-7B")).toBe("hf--Qwen--Qwen2.5-Coder-7B");
  });

  it("builds stable id from pollinations model name", () => {
    expect(rackIdFromPollinations("flux-pro")).toBe("pollinations-flux-pro");
  });

  it("summarizes rack counts by source", () => {
    const rack = mergeWithBuiltinRack([
      rackEntryFromHfHit({
        modelId: "test/free-model",
        status: "WORKING",
        accessMode: "FREE",
      }),
      pollinationsEntry("flux-anime"),
    ]);
    const counts = summarizeRackCounts(rack);
    expect(counts.builtin).toBe(1);
    expect(counts.hf).toBe(1);
    expect(counts.spaces).toBe(0);
    expect(counts.cloud).toBeGreaterThanOrEqual(1);
  });

  it("tags entries for picker UI", () => {
    expect(rackEntryTagLabel({ source: "builtin", adapter: "gemma-local" } as never)).toBe("מובנה");
    expect(rackEntryTagLabel({ adapter: "hf-local-text", status: "not_downloaded" } as never)).toBe(
      "לא הורד",
    );
    expect(rackEntryTagLabel({ adapter: "hf-local-text", status: "ready" } as never)).toBe("מוכן");
    expect(rackEntryTagLabel({ source: "cloud-scan" } as never)).toBeNull();
  });

  it("classifies hub pipelines into rack modalities", () => {
    expect(modalityFromPipeline("text-generation", "meta/llama")).toBe("text");
    expect(modalityFromPipeline("text-to-speech", "org/tts")).toBe("audio");
    expect(modalityFromPipeline("image-to-text", "org/blip")).toBe("vision");
    expect(modalityFromPipeline("text-generation", "Qwen/Qwen2.5-Coder-7B")).toBe("code");
  });

  it("maps WORKING FREE hf hit to ready", () => {
    const entry = rackEntryFromHfHit({
      modelId: "stabilityai/stable-diffusion-xl-base-1.0",
      category: "Image",
      pipelineTag: "text-to-image",
      status: "WORKING",
      accessMode: "FREE",
    });
    expect(entry.modality).toBe("image");
    expect(entry.status).toBe("ready");
    expect(entry.hfAccessMode).toBe("FREE");
  });

  it("maps WORKING TOKEN hf hit to token_required (not in picker)", () => {
    const entry = rackEntryFromHfHit({
      modelId: "Qwen/Qwen2.5-0.5B-Instruct",
      pipelineTag: "text-generation",
      status: "WORKING",
      accessMode: "TOKEN",
    });
    expect(entry.status).toBe("token_required");
    expect(pickableRackModels([entry])).toHaveLength(0);
  });

  it("maps PROVIDER REQUIRED to token_required and excludes from picker", () => {
    const entry = rackEntryFromHfHit({
      modelId: "org/model",
      category: "Code",
      status: "PROVIDER REQUIRED",
      accessMode: "TOKEN",
    });
    expect(entry.status).toBe("token_required");
    expect(pickableRackModels([entry])).toHaveLength(0);
  });

  it("upsertFree ignores token_required entries", () => {
    const tokenEntry = rackEntryFromHfHit({
      modelId: "org/needs-token",
      status: "PROVIDER REQUIRED",
      accessMode: "TOKEN",
    });
    const rack = upsertFreeRackModel(tokenEntry);
    expect(rack.some((r) => r.hfModelId === "org/needs-token")).toBe(false);
  });

  it("drops stored hf-scan entries that require token on load", () => {
    const tokenEntry = rackEntryFromHfHit({
      modelId: "org/needs-token",
      status: "WORKING",
      accessMode: "TOKEN",
    });
    tokenEntry.status = "ready";
    tokenEntry.hfAccessMode = "TOKEN";
    const merged = mergeWithBuiltinRack([tokenEntry]);
    expect(merged.some((r) => r.hfModelId === "org/needs-token")).toBe(false);
  });

  it("merge keeps only pickable scanned models", () => {
    const free = rackEntryFromHfHit({
      modelId: "test/free-model",
      status: "WORKING",
      accessMode: "FREE",
      category: "Code",
    });
    const paid = rackEntryFromHfHit({
      modelId: "test/paid-model",
      status: "PROVIDER REQUIRED",
      accessMode: "TOKEN",
      category: "Code",
    });
    const merged = mergeWithBuiltinRack([free, paid]);
    expect(merged.some((r) => r.hfModelId === "test/free-model")).toBe(true);
    expect(merged.some((r) => r.hfModelId === "test/paid-model")).toBe(false);
  });
});
