import { describe, expect, it, vi, beforeEach } from "vitest";

const translateTextsMock = vi.fn();

vi.mock("../groveeNews/engine/translate/googleTranslate", () => ({
  translateTexts: (...args: unknown[]) => translateTextsMock(...args),
}));

import {
  buildLocalTextHistoryForModel,
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
      .mockResolvedValueOnce({ texts: ["Hello"], provider: "cache" })
      .mockResolvedValueOnce({ texts: ["Hey"], provider: "cache" });

    const prepared = await prepareLocalTextTurnForModel(
      "מה שלומך?",
      [
        { role: "user", content: "שלום" },
        { role: "assistant", content: "היי" },
      ],
      "he",
    );

    expect(prepared.prompt).toBe("Hi");
    expect(prepared.history[0].content).toBe("Hello");
    expect(prepared.history[1].content).toBe("Hey");
    expect(prepared.systemPrompt).toContain("GROVEE");
  });

  it("falls back to original text when translate fails", async () => {
    translateTextsMock.mockRejectedValue(new Error("offline"));
    const history = await translateLocalTextHistoryForModel(
      [{ role: "user", content: "שלום" }],
      "he",
    );
    expect(history[0].content).toBe("שלום");
  });

  it("buildLocalTextHistoryForModel pins remember messages mid-chat", () => {
    const entries = Array.from({ length: 10 }, (_, i) => ({
      role: (i % 2 === 0 ? "user" : "assistant") as const,
      content: `msg-${i}`,
    }));
    entries[6] = { role: "user", content: "זכור: העיר האהובה עליי היא פריז" };
    entries[7] = { role: "assistant", content: "Noted." };
    const pinned = [6, 7];
    const { entries: picked, sourceIndices } = buildLocalTextHistoryForModel(entries, {
      maxMessageSlots: 6,
      pinnedSourceIndices: pinned,
    });
    expect(sourceIndices).toContain(6);
    expect(picked.some((e) => /פריז/.test(e.content))).toBe(true);
  });

  it("QA memory script: Paris survives filler turns", () => {
    const entries = [
      { role: "user" as const, content: "remember: my favorite city is Paris" },
      { role: "assistant" as const, content: "Got it." },
      { role: "user" as const, content: "weather?" },
      { role: "assistant" as const, content: "Sunny." },
      { role: "user" as const, content: "random fact" },
      { role: "assistant" as const, content: "Cats sleep a lot." },
      { role: "user" as const, content: "which city do I love?" },
    ];
    const { entries: picked } = buildLocalTextHistoryForModel(entries, {
      maxMessageSlots: 12,
      pinnedSourceIndices: [0, 1],
    });
    expect(picked.some((e) => /paris/i.test(e.content))).toBe(true);
  });
});
