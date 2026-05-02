import { describe, expect, it } from "vitest";
import {
  cleanEnglishImagePrompt,
  isCodeRequest,
  isImageGenerationRequest,
  isRtlText,
  isSimpleGreeting,
} from "./chatIntents";

describe("chatIntents", () => {
  it("detects Hebrew image requests", () => {
    expect(isImageGenerationRequest("צור תמונה של חתול")).toBe(true);
    expect(isImageGenerationRequest("מה השעה?")).toBe(false);
  });

  it("detects code requests (Hebrew + English)", () => {
    expect(isCodeRequest("כתוב פונקציה בפייתון")).toBe(true);
    expect(isCodeRequest("fix this TypeScript error")).toBe(true);
    expect(isCodeRequest("ספר לי בדיחה")).toBe(false);
  });

  it("normalizes English image prompt to one line", () => {
    expect(cleanEnglishImagePrompt('  "a red apple on a table"  \nmore')).toContain("red apple");
    expect(cleanEnglishImagePrompt("")).toBe("high quality detailed scene");
  });

  it("RTL heuristic", () => {
    expect(isRtlText("שלום")).toBe(true);
    expect(isRtlText("hello")).toBe(false);
  });

  it("simple greetings", () => {
    expect(isSimpleGreeting("היי")).toBe(true);
    expect(isSimpleGreeting("hello")).toBe(true);
    expect(isSimpleGreeting("צור תמונה")).toBe(false);
  });
});
