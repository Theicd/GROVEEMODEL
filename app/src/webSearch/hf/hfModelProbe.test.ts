import { describe, expect, it } from "vitest";
import { classifyModelStatus, detectInferenceProvider } from "./hfModelProbe";

describe("hfModelProbe", () => {
  it("classifies 200 as WORKING", () => {
    expect(classifyModelStatus(200, '{"choices":[{"message":{"content":"OK"}}]}')).toBe("WORKING");
  });

  it("classifies token required from body", () => {
    expect(classifyModelStatus(403, "Please pass a hf_token")).toBe("PROVIDER REQUIRED");
  });

  it("detects HF inference provider", () => {
    expect(detectInferenceProvider("https://router.huggingface.co/v1/chat/completions", {})).toBe(
      "HF inference",
    );
  });

  it("detects Together provider", () => {
    expect(detectInferenceProvider("https://api.together.xyz/v1/chat", {})).toBe("Together");
  });
});
