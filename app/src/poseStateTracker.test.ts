import { describe, expect, it } from "vitest";
import { PoseStateTracker } from "./poseStateTracker";

describe("poseStateTracker", () => {
  it("commits after consecutive agreeing frames", () => {
    const tracker = new PoseStateTracker();
    expect(tracker.observe({ poseState: "standing", confidence: 0.7 }).committed).toBe(false);
    const second = tracker.observe({ poseState: "standing", confidence: 0.72 });
    expect(second.committed).toBe(true);
    expect(tracker.committedState).toBe("standing");
  });

  it("emits pose_change when committed state shifts", () => {
    const tracker = new PoseStateTracker();
    tracker.observe({ poseState: "sitting", confidence: 0.8 });
    tracker.observe({ poseState: "sitting", confidence: 0.82 });
    const change = tracker.observe({ poseState: "standing", confidence: 0.85 });
    expect(change.committed).toBe(false);
    const commit = tracker.observe({ poseState: "standing", confidence: 0.86 });
    expect(commit.changed).toBe(true);
    expect(commit.from).toBe("sitting");
    expect(commit.to).toBe("standing");
  });
});
