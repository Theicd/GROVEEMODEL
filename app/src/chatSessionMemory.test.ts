import { describe, expect, it } from "vitest";
import {
  answerSessionMemoryRecall,
  collectSessionMemoryFacts,
  extractCityFromMemoryFact,
  extractMemoryFactFromSave,
  isUserMemoryRecallQuery,
  isUserMemorySaveRequest,
  memoryPinnedSourceIndices,
} from "./chatSessionMemory";

describe("chatSessionMemory", () => {
  it("extracts save facts", () => {
    expect(extractMemoryFactFromSave("זכור: העיר האהובה עליי היא פריז")).toBe(
      "העיר האהובה עליי היא פריז",
    );
    expect(isUserMemorySaveRequest("remember: my favorite city is Paris")).toBe(true);
  });

  it("collects facts from message list", () => {
    const facts = collectSessionMemoryFacts([
      { role: "user", content: "מה מזג האוויר?" },
      { role: "assistant", content: "חם." },
      { role: "user", content: "זכור: העיר האהובה עליי היא פריז" },
    ]);
    expect(facts).toEqual(["העיר האהובה עליי היא פריז"]);
  });

  it("extracts city from Hebrew fact", () => {
    expect(extractCityFromMemoryFact("העיר האהובה עליי היא פריז")).toBe("פריז");
  });

  it("answers recall for favorite city (QA script)", () => {
    const facts = collectSessionMemoryFacts([
      { role: "user", content: "זכור: העיר האהובה עליי היא פריז" },
    ]);
    expect(isUserMemoryRecallQuery("מה העיר האהובה עלי?")).toBe(true);
    const reply = answerSessionMemoryRecall("מה העיר האהובה עלי?", facts, "he");
    expect(reply).toContain("פריז");
  });

  it("pins memory message indices", () => {
    const entries = [
      { role: "user", content: "שלום" },
      { role: "assistant", content: "היי" },
      { role: "user", content: "זכור: העיר האהובה עליי היא פריז" },
      { role: "assistant", content: "בסדר" },
    ];
    expect(memoryPinnedSourceIndices(entries)).toEqual([2, 3]);
  });
});
