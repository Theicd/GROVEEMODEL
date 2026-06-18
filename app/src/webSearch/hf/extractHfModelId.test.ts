import { describe, expect, it } from "vitest";
import { extractHfModelIdFromQuery } from "./extractHfModelId";

describe("extractHfModelIdFromQuery", () => {
  it("parses HF URL", () => {
    expect(extractHfModelIdFromQuery("https://huggingface.co/meta-llama/Llama-3.2-1B")).toBe(
      "meta-llama/Llama-3.2-1B",
    );
  });

  it("parses org/model slug", () => {
    expect(extractHfModelIdFromQuery("huggingface Qwen/Qwen2.5-7B-Instruct")).toBe(
      "Qwen/Qwen2.5-7B-Instruct",
    );
  });

  it("returns null for generic query", () => {
    expect(extractHfModelIdFromQuery("stable diffusion models")).toBeNull();
  });
});
