import { describe, expect, it } from "vitest";
import { movenetKeypointsToCoco } from "./moveNetLoader";

describe("moveNetLoader", () => {
  it("maps MoveNet keypoints to COCO-17 array", () => {
    const raw = Array.from({ length: 17 }, (_, i) => ({
      x: i * 10,
      y: i * 5,
      score: 0.8,
    }));
    const kps = movenetKeypointsToCoco(raw);
    expect(kps).toHaveLength(17);
    expect(kps[9]).toEqual({ x: 90, y: 45, score: 0.8 });
  });
});
