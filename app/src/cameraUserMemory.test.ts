import { describe, expect, it } from "vitest";
import {
  applyProfileHints,
  buildRollingSummary,
  buildUserMemoryPromptBlock,
  extractProfileHintsFromUserText,
  searchCameraHistory,
} from "./cameraUserMemory";
import { emptyUserProfile } from "./cameraUserMemory";
import type { CameraMessage } from "./cameraSession";

describe("cameraUserMemory", () => {
  it("extracts sci-fi hobby from user text", () => {
    const hints = extractProfileHintsFromUserText("אני מחפש רעיון לסיפור מדע בדיוני");
    expect(hints.hobbies).toContain("מדע בדיוני");
    expect(hints.hobbies).toContain("סיפורים");
  });

  it("merges profile hints", () => {
    const p = applyProfileHints(emptyUserProfile(), {
      name: "יניב",
      hobbies: ["מדע בדיוני"],
    });
    expect(p.name).toBe("יניב");
    expect(p.hobbies).toContain("מדע בדיוני");
  });

  it("searches history by keyword", () => {
    const messages: CameraMessage[] = [
      {
        id: "1",
        role: "user",
        kind: "user",
        content: "בוא נדבר על ברמודה",
        ts: 1,
      },
      {
        id: "2",
        role: "assistant",
        kind: "reply",
        content: "רעיון מעניין לסיפור",
        ts: 2,
      },
    ];
    const hits = searchCameraHistory(messages, "ברמודה");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].snippet).toMatch(/ברמודה/);
  });

  it("builds rolling summary from recent turns", () => {
    const messages: CameraMessage[] = [
      { id: "1", role: "user", kind: "user", content: "שלום", ts: 1 },
      { id: "2", role: "assistant", kind: "reply", content: "היי!", ts: 2 },
    ];
    const s = buildRollingSummary(messages);
    expect(s).toMatch(/User: שלום/);
    expect(s).toMatch(/HAL: היי/);
  });

  it("builds memory prompt block", () => {
    const block = buildUserMemoryPromptBlock({
      profile: { name: "יניב", hobbies: ["מדע בדיוני"], notes: "", updatedAt: 1 },
      rollingSummary: "User: שלום\nHAL: היי",
    });
    expect(block).toMatch(/name=יניב/);
    expect(block).toMatch(/hobbies=מדע בדיוני/);
  });
});
