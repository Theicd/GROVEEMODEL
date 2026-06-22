// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getSearchCompanionUrl,
  isLocalOpenSerpUrl,
  resolveSearchCompanionFetchBase,
  setSearchCompanionUrl,
  usesDevOpenSerpProxy,
} from "./companionSettings";

describe("companionSettings", () => {
  beforeEach(() => {
    localStorage.clear();
    setSearchCompanionUrl("");
    vi.stubGlobal("location", { origin: "http://127.0.0.1:5180" });
  });

  it("detects local OpenSERP URLs", () => {
    expect(isLocalOpenSerpUrl("http://127.0.0.1:7000")).toBe(true);
    expect(isLocalOpenSerpUrl("http://localhost:7000/")).toBe(true);
    expect(isLocalOpenSerpUrl("https://search.example.com")).toBe(false);
  });

  it("routes local companion through dev proxy", () => {
    setSearchCompanionUrl("http://127.0.0.1:7000");
    expect(getSearchCompanionUrl()).toBe("http://127.0.0.1:7000");
    expect(resolveSearchCompanionFetchBase()).toBe("http://127.0.0.1:5180/api/openserp");
    expect(usesDevOpenSerpProxy()).toBe(true);
  });

  it("uses saved remote URL when not localhost", () => {
    setSearchCompanionUrl("https://openserp.home.lan:7000");
    expect(resolveSearchCompanionFetchBase()).toBe("https://openserp.home.lan:7000");
    expect(usesDevOpenSerpProxy()).toBe(false);
  });
});
