import { describe, expect, it } from "vitest";
import {
  appendLiveCaption,
  chunkAudioMetrics,
  cleanTranscript,
  diffChunkWords,
  isHallucinatedTranscript,
  shouldAcceptTranscript,
  trimCaptionWords,
} from "./liveCaptionRoll";

describe("liveCaptionRoll", () => {
  it("merges only novel tail words from overlapping chunks", () => {
    const a = diffChunkWords("they own the neighborhood", "own the neighborhood I sat");
    expect(a).toBe("I sat");
  });

  it("keeps only the last N words on screen", () => {
    const long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen";
    expect(trimCaptionWords(long, 8)).toBe("eight nine ten eleven twelve thirteen fourteen fifteen");
  });

  it("rejects whisper hallucinations on silence", () => {
    expect(isHallucinatedTranscript("you")).toBe(true);
    expect(isHallucinatedTranscript("thank you")).toBe(true);
    expect(isHallucinatedTranscript("they own the neighborhood")).toBe(false);
  });

  it("cleans punctuation junk from transcript", () => {
    expect(cleanTranscript("... ,they own the neighborhood.")).toBe("they own the neighborhood");
  });

  it("rejects duplicate chunk retranscription", () => {
    const metrics = chunkAudioMetrics(new Float32Array([0.02, -0.02, 0.03]));
    expect(shouldAcceptTranscript("hello world", "hello world", "hello world", metrics)).toBe(false);
  });

  it("appends accepted words and tracks last chunk", () => {
    const metrics = { rms: 0.03, zcr: 0.05 };
    const first = appendLiveCaption("", "", "they own the neighborhood", metrics);
    expect(first.accepted).toBe(true);
    const second = appendLiveCaption(first.roll, first.lastChunk, "neighborhood I sat down", metrics);
    expect(second.roll).toBe("they own the neighborhood I sat down");
  });
});
