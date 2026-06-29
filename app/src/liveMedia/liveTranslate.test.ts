import { describe, expect, it } from "vitest";
import { resampleTo16kMono } from "./liveAudioCapture";
import {
  broadcastLangToSpeechCode,
  broadcastLangToWhisperLanguage,
  CAPTION_TARGET_NONE,
  shouldTranslateCaptions,
  speechLangToWhisperLanguage,
} from "./liveTranslate";

describe("liveTranslate", () => {
  it("maps broadcast language codes to speech recognition locales", () => {
    expect(broadcastLangToSpeechCode("heb")).toBe("he-IL");
    expect(broadcastLangToSpeechCode("eng")).toBe("en-US");
    expect(broadcastLangToSpeechCode("rus")).toBe("ru-RU");
    expect(broadcastLangToSpeechCode(undefined)).toBe("en-US");
  });

  it("maps speech locales to whisper language tokens", () => {
    expect(speechLangToWhisperLanguage("he-IL")).toBe("hebrew");
    expect(speechLangToWhisperLanguage("en-US")).toBe("english");
    expect(broadcastLangToWhisperLanguage("heb")).toBe("hebrew");
  });

  it("skips translation for none or matching source/target", () => {
    expect(shouldTranslateCaptions("en-US", CAPTION_TARGET_NONE)).toBe(false);
    expect(shouldTranslateCaptions("en-US", "en")).toBe(false);
    expect(shouldTranslateCaptions("he-IL", "he")).toBe(false);
    expect(shouldTranslateCaptions("en-US", "he")).toBe(true);
    expect(shouldTranslateCaptions("he-IL", "en")).toBe(true);
  });
});

describe("liveAudioCapture", () => {
  it("resamples stereo-rate buffers down to 16 kHz", () => {
    const input = new Float32Array(48000);
    for (let i = 0; i < input.length; i += 1) input[i] = Math.sin(i / 100);
    const out = resampleTo16kMono(input, 48000);
    expect(out.length).toBe(16000);
  });
});
