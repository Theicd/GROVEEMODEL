// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { readTvDeepLink } from "./deepLinks";

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
});
