import { describe, expect, it } from "vitest";
import {
  cleanEnglishImagePrompt,
  isCodeRequest,
  isImageGenerationRequest,
  isRtlText,
  isSimpleGreeting,
  stripImageEchoes,
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

describe("stripImageEchoes", () => {
  it("removes markdown image tags", () => {
    expect(stripImageEchoes("hello\n![alt](https://x/y.jpg)\nworld")).toBe("hello\nworld");
  });

  it("removes bare and angle-bracketed URLs from the text", () => {
    const out1 = stripImageEchoes("see https://image.pollinations.ai/prompt/x?y=1 thanks");
    expect(out1).not.toContain("http");
    expect(out1).toContain("see");
    expect(out1).toContain("thanks");
    expect(
      stripImageEchoes(
        "<https://image.pollinations.ai/?prompt=A%0Aphotorealistic+image+of+a+cute+golden+retriever?width=",
      ),
    ).toBe("");
  });

  it("removes 'URL:/width=/height=/model=' lines that the model copies from the prompt", () => {
    const messy = [
      "נוצרה תמונה ריאליסטית של כלבת גולדן רטריבר.",
      "URL:",
      "<https://image.pollinations.ai/?prompt=A%0Aphotorealistic+image",
      "width=1000",
      "height=1600",
      "model=flux",
      "nologo=true",
    ].join("\n");
    expect(stripImageEchoes(messy)).toBe("נוצרה תמונה ריאליסטית של כלבת גולדן רטריבר.");
  });

  it("collapses consecutive blank lines and trims edges", () => {
    expect(stripImageEchoes("\n\n hello \n\n\n world \n\n")).toBe("hello\nworld");
  });

  it("leaves clean caption text untouched", () => {
    const clean = "נוצרה תמונה ריאליסטית של כלבת גולדן רטריבר חמודה משחקת בפארק שמשי.";
    expect(stripImageEchoes(clean)).toBe(clean);
  });
});
