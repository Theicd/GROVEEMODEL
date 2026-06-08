import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dirname, "../..");

describe("vision always-on policy", () => {
  it("VisionPipeline does not gate YOLO on heavyPaused", () => {
    const src = readFileSync(join(ROOT, "app/src/vision-lab/core/VisionPipeline.ts"), "utf8");
    expect(src).not.toContain("heavyPaused");
    expect(src).not.toMatch(/if\s*\(\s*!this\.heavyPaused/);
  });

  it("GroveeVisionRunner does not pause YOLO during deep vision or chat", () => {
    const src = readFileSync(join(ROOT, "app/src/GroveeVisionRunner.ts"), "utf8");
    expect(src).not.toContain("setHeavyPaused");
    expect(src).not.toMatch(/pipeline\.stop\(\)[\s\S]{0,80}document\.hidden/);
  });

  it("face-api model shards exist under public/models/face-api", () => {
    const dir = join(ROOT, "public/models/face-api");
    const names = [
      "tiny_face_detector_model-shard1",
      "face_landmark_68_model-shard1",
      "face_expression_model-shard1",
      "age_gender_model-shard1",
    ];
    for (const name of names) {
      expect(() => readFileSync(join(dir, name))).not.toThrow();
    }
  });
});
