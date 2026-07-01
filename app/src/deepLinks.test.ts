// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { readSuperSportDeepLink, readTvDeepLink } from "./deepLinks";

describe("readTvDeepLink", () => {
  afterEach(() => {
    window.history.replaceState({}, "", "/");
  });

  it("returns false for default URL", () => {
    expect(readTvDeepLink()).toBe(false);
  });

  it("detects ?tv=1", () => {
    window.history.replaceState({}, "", "/?tv=1");
    expect(readTvDeepLink()).toBe(true);
  });

  it("detects #tv hash", () => {
    window.history.replaceState({}, "", "/#tv");
    expect(readTvDeepLink()).toBe(true);
  });

  it("treats SUPER SPORT deep link as a TV deep link", () => {
    window.history.replaceState({}, "", "/?sport=1");
    expect(readTvDeepLink()).toBe(true);
  });
});

describe("readSuperSportDeepLink", () => {
  afterEach(() => {
    window.history.replaceState({}, "", "/");
  });

  it("returns false for default URL", () => {
    expect(readSuperSportDeepLink()).toBe(false);
  });

  it("returns false for plain ?tv=1", () => {
    window.history.replaceState({}, "", "/?tv=1");
    expect(readSuperSportDeepLink()).toBe(false);
  });

  it("detects ?sport=1", () => {
    window.history.replaceState({}, "", "/?sport=1");
    expect(readSuperSportDeepLink()).toBe(true);
  });

  it("detects #supersport hash", () => {
    window.history.replaceState({}, "", "/#supersport");
    expect(readSuperSportDeepLink()).toBe(true);
  });
});
