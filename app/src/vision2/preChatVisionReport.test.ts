import { describe, expect, it } from "vitest";
import { buildPreChatVisionReport, resolvePersonVisibleForChat } from "./preChatVisionReport";
import { WorldMemory } from "../worldMemory";
import type { VisionResult } from "../vision-lab/core/types";

const personVision = (conf = 0.61) =>
  ({
    objects: [
      {
        label: "person",
        displayLabel: "person",
        confidence: conf,
        bbox: { x: 0, y: 0, width: 0.3, height: 0.5 },
      },
    ],
    faces: [],
    interactions: [],
    bodyLanguage: [],
  }) as unknown as VisionResult;

describe("preChatVisionReport", () => {
  it("marks person visible from YOLO even when consciousness unstable", () => {
    const world = new WorldMemory();
    world.personPresent = false;
    const vision = personVision(0.61);
    const visible = resolvePersonVisibleForChat({
      vision,
      dialogue: {
        consciousness: {
          soul: "PHANTOM_DETECTION",
          personStable: false,
          rawDetected: true,
          confidence: 0.61,
          gemmaBlock: "Person in frame: NO (authoritative)",
        },
      } as never,
      world,
    });
    expect(visible).toBe(true);
    const report = buildPreChatVisionReport({
      vision,
      dialogue: null,
      world,
      cameraActive: true,
      snapshotAttached: true,
    });
    expect(report.text).toMatch(/personVisible=yes/);
    expect(report.internalEn).toMatch(/Person visible: YES/);
    expect(report.internalEn).toMatch(/YOLO persons: 1/);
  });

  it("includes holding and face data for demographics", () => {
    const world = new WorldMemory();
    world.holding = ["cell phone"];
    const vision = {
      ...personVision(),
      faces: [
        {
          id: 1,
          estimatedAge: 55,
          estimatedGender: "Male",
          gazeDirection: "Center",
          bbox: { x: 0, y: 0, width: 0.1, height: 0.1 },
        },
      ],
    } as unknown as VisionResult;
    const report = buildPreChatVisionReport({
      vision,
      dialogue: null,
      world,
      cameraActive: true,
    });
    expect(report.internalEn).toMatch(/Holding \(sensor\): cell phone/);
    expect(report.internalEn).toMatch(/male \(estimate\)/);
    expect(report.internalEn).toMatch(/age ~44/);
    expect(report.text).toMatch(/holding=cell phone/);
  });
});
