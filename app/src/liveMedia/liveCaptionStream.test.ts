import { describe, expect, it } from "vitest";
import { novelTimedWords } from "./liveCaptionStream";

describe("liveCaptionStream", () => {
  it("returns only novel timed words after overlap", () => {
    const prev = "they own the neighborhood";
    const words = [
      { text: "neighborhood", start: 0.8, end: 1.2 },
      { text: "I", start: 1.2, end: 1.3 },
      { text: "sat", start: 1.3, end: 1.6 },
    ];
    const novel = novelTimedWords(prev, words);
    expect(novel.map((w) => w.text).join(" ")).toBe("I sat");
  });
});
