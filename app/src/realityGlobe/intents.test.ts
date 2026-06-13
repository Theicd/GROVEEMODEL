import { describe, expect, it } from "vitest";
import { shouldOpenGlobePanel, isGlobePresentationQuery } from "./intents";

describe("shouldOpenGlobePanel", () => {
  it("does not open globe for weather-only query", () => {
    expect(shouldOpenGlobePanel("מה מזג האוויר בתל אביב", ["weather"])).toBe(false);
  });

  it("does not open globe for worldtime-only query", () => {
    expect(shouldOpenGlobePanel("מה השעה בישראל", ["worldtime"])).toBe(false);
  });

  it("opens globe for explicit map request", () => {
    expect(shouldOpenGlobePanel("הצג על המפה את berlin", ["places"])).toBe(true);
    expect(isGlobePresentationQuery("הצג על המפה את berlin")).toBe(true);
  });

  it("opens globe for earthquake intent with keyword", () => {
    expect(shouldOpenGlobePanel("רעידות אדמה", ["earthquake"])).toBe(false);
    expect(shouldOpenGlobePanel("הצג על המפה רעידות אדמה", ["earthquake"])).toBe(true);
  });

  it("does not open globe for aviation count without map request", () => {
    expect(shouldOpenGlobePanel("כמה מטוסים נמצאים כרגע מעל ישראל?", ["aviation"])).toBe(false);
  });

  it("does not open globe for ships without map request", () => {
    expect(shouldOpenGlobePanel("אילו ספינות בתעלת סואץ", ["ships"])).toBe(false);
  });

  it("does not open globe for satellite count without map request", () => {
    expect(shouldOpenGlobePanel("כמה לוויינים פעילים", ["satellite"])).toBe(false);
  });
});
