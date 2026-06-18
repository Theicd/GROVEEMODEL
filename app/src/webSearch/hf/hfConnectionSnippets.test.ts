import { describe, expect, it } from "vitest";
import { buildHfCurlSnippet, buildHfPythonSnippet, statusBadgeLabel } from "./hfConnectionSnippets";

describe("hfConnectionSnippets", () => {
  it("builds curl with model id", () => {
    const curl = buildHfCurlSnippet("meta-llama/Llama-3.2-1B-Instruct");
    expect(curl).toContain("curl ");
    expect(curl).toContain("meta-llama/Llama-3.2-1B-Instruct");
    expect(curl).toContain("HF_TOKEN");
  });

  it("builds python snippet with model id", () => {
    const py = buildHfPythonSnippet("Qwen/Qwen2.5-7B-Instruct");
    expect(py).toContain("import requests");
    expect(py).toContain("Qwen/Qwen2.5-7B-Instruct");
  });

  it("labels WORKING status in Hebrew", () => {
    expect(statusBadgeLabel("WORKING", "he")).toMatch(/עובד/);
  });
});
