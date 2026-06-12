import { describe, expect, it } from "vitest";
import {
  appendTopicToLog,
  buildCameraHistoryForWorker,
  defaultCameraSessionStore,
  loadCameraSessionStore,
} from "./cameraSession";

describe("cameraSession", () => {
  it("builds worker history from camera messages only", () => {
    const store = defaultCameraSessionStore();
    store.messages = [
      { id: "1", role: "user", kind: "user", content: "מה אתה רואה?", ts: 1 },
      { id: "2", role: "assistant", kind: "proactive", content: "שמתי לב אליך", ts: 2 },
    ];
    const history = buildCameraHistoryForWorker(store.messages);
    expect(history).toHaveLength(2);
    expect(history[0].role).toBe("user");
    expect(history[1].content).toContain("שמתי לב");
  });

  it("appendTopicToLog dedupes and caps", () => {
    let log = appendTopicToLog([], "visibility");
    log = appendTopicToLog(log, "mood");
    log = appendTopicToLog(log, "visibility");
    expect(log).toEqual(["mood", "visibility"]);
  });

  it("loadCameraSessionStore returns defaults when empty", () => {
    const s = loadCameraSessionStore();
    expect(s.version).toBe(2);
    expect(s.profile).toBeDefined();
    expect(Array.isArray(s.messages)).toBe(true);
  });
});
