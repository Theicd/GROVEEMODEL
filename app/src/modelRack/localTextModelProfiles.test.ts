import { describe, expect, it } from "vitest";
import {
  HUNYUAN_HF_MODEL_ID,
  HUNYUAN_INTERIM_ONNX_HF_MODEL_ID,
  HUNYUAN_RACK_ID,
  SMOLLM_HF_MODEL_ID,
  resolveLocalTextLoadModelId,
} from "./localTextModels";
import {
  isHunyuanLocalTextModel,
  localTextProfileForHfModelId,
  localTextProfileForRackId,
  localTextSettingsForProfile,
  resolveHistoryCharBudget,
} from "./localTextModelProfiles";

describe("localTextModelProfiles", () => {
  it("resolves Hunyuan profile by rack id", () => {
    const profile = localTextProfileForRackId(HUNYUAN_RACK_ID);
    expect(profile?.hfModelId).toBe(HUNYUAN_HF_MODEL_ID);
    expect(profile?.historyCharBudget).toBe(64_000);
    expect(profile?.settingsOverrides.historyTurns).toBe(40);
  });

  it("falls back to SmolLM profile for unknown hf id", () => {
    const profile = localTextProfileForHfModelId(SMOLLM_HF_MODEL_ID);
    expect(profile.rackId).toContain("SmolLM2-360M");
    expect(profile.historyCharBudget).toBe(3000);
  });

  it("merges Hunyuan settings overrides", () => {
    const profile = localTextProfileForRackId(HUNYUAN_RACK_ID)!;
    const settings = localTextSettingsForProfile(profile);
    expect(settings.temperature).toBe(0.7);
    expect(settings.historyTurns).toBe(40);
    expect(settings.maxNewTokens).toBe(384);
  });

  it("detects Hunyuan hf model id", () => {
    expect(isHunyuanLocalTextModel(HUNYUAN_HF_MODEL_ID)).toBe(true);
    expect(isHunyuanLocalTextModel(SMOLLM_HF_MODEL_ID)).toBe(false);
  });

  it("maps Hunyuan rack id to interim ONNX load target", () => {
    expect(resolveLocalTextLoadModelId(HUNYUAN_HF_MODEL_ID)).toBe(HUNYUAN_INTERIM_ONNX_HF_MODEL_ID);
    expect(resolveLocalTextLoadModelId(SMOLLM_HF_MODEL_ID)).toBe(SMOLLM_HF_MODEL_ID);
  });

  it("resolveHistoryCharBudget caps on low device memory", () => {
    const profile = localTextProfileForRackId(HUNYUAN_RACK_ID)!;
    expect(resolveHistoryCharBudget(profile)).toBe(64_000);
  });
});
