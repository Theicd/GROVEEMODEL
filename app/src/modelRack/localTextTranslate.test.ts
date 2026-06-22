import { describe, expect, it, vi, beforeEach } from "vitest";

const translateTextsMock = vi.fn();

vi.mock("../groveeNews/engine/translate/googleTranslate", () => ({
  translateTexts: (...args: unknown[]) => translateTextsMock(...args),
}));

import {
  localTextToModelLanguage,
  localTextToUiLanguage,
  needsLocalTextTranslationBridge,
  prepareLocalTextTurnForModel,
  translateLocalTextHistoryForModel,
} from "./localTextTranslate";

describe("localTextTranslate", () => {
  beforeEach(() => {
    translateTextsMock.mockReset();
  });

  it("skips bridge when UI is English", () => {
    expect(needsLocalTextTranslationBridge("en")).toBe(false);
  });

  it("enables bridge when UI is Hebrew", () => {
    expect(needsLocalTextTranslationBridge("he")).toBe(true);
  });

  it("translates prompt he→en for Hebrew UI", async () => {
    translateTextsMock.mockResolvedValue({ texts: ["Hello"], provider: "cache" });
    const out = await localTextToModelLanguage("שלום", "he");
    expect(out).toBe("Hello");
    expect(translateTextsMock).toHaveBeenCalledWith(["שלום"], "en", "he");
  });

  it("passes through English UI text unchanged", async () => {
    const out = await localTextToModelLanguage("Hello", "en");
    expect(out).toBe("Hello");
    expect(translateTextsMock).not.toHaveBeenCalled();
  });

  it("translates reply en→he for Hebrew UI", async () => {
    translateTextsMock.mockResolvedValue({ texts: ["שלום"], provider: "cache" });
    const out = await localTextToUiLanguage("Hello", "he");
    expect(out).toBe("שלום");
    expect(translateTextsMock).toHaveBeenCalledWith(["Hello"], "he", "en");
  });

  it("prepareLocalTextTurnForModel batches history translation", async () => {
    translateTextsMock
      .mockResolvedValueOnce({ texts: ["Hi"], provider: "cache" })
      .mockResolvedValueOnce({ texts: ["How are you?", "I am fine"], provider: "cache" });

    const prepared = await prepareLocalTextTurnForModel(
      "מה שלומך?",
      [
        { role: "user", content: "שלום" },
        { role: "assistant", content: "היי" },
      ],
      "he",
    );

    expect(prepared.prompt).toBe("Hi");
    expect(prepared.history[0].content).toBe("How are you?");
    expect(prepared.history[1].content).toBe("I am fine");
    expect(prepared.systemPrompt).toContain("English only");
  });

  it("falls back to original text when translate fails", async () => {
    translateTextsMock.mockRejectedValue(new Error("offline"));
    const history = await translateLocalTextHistoryForModel(
      [{ role: "user", content: "שלום" }],
      "he",
    );
    expect(history[0].content).toBe("שלום");
  });
});
