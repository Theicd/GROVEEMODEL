import { describe, expect, it } from "vitest";
import {
  IMAGE_MODEL_OPTIONS,
  buildPollinationsUrl,
  normalizePollinationsModel,
} from "./cloudImage";

/**
 * The chat flow renders this URL straight into a markdown image: ![Generated](URL).
 * Bad URLs (unencoded quotes, missing model, empty prompt) silently produce a broken
 * <img>. These tests pin the format so we don't regress that.
 */

describe("cloudImage — buildPollinationsUrl", () => {
  it("produces a valid HTTPS Pollinations URL with the expected base path", () => {
    const u = buildPollinationsUrl({ prompt: "a red cat" });
    expect(u.startsWith("https://image.pollinations.ai/prompt/")).toBe(true);
    const parsed = new URL(u);
    expect(parsed.protocol).toBe("https:");
    expect(parsed.host).toBe("image.pollinations.ai");
    expect(parsed.pathname.startsWith("/prompt/")).toBe(true);
  });

  it("encodes Hebrew, quotes, slashes, '&' and emoji safely", () => {
    const dirty = 'חתול חמוד/"a fluffy" cat & dog 🐱';
    const u = buildPollinationsUrl({ prompt: dirty });
    expect(u).not.toMatch(/\s/); // no raw spaces
    expect(u).not.toContain('"'); // no raw quotes
    const promptPart = decodeURIComponent(new URL(u).pathname.replace("/prompt/", ""));
    expect(promptPart).toBe(dirty);
  });

  it("defaults to flux + 1024x1024 + nologo when not provided", () => {
    const u = new URL(buildPollinationsUrl({ prompt: "ok" }));
    expect(u.searchParams.get("model")).toBe("flux");
    expect(u.searchParams.get("width")).toBe("1024");
    expect(u.searchParams.get("height")).toBe("1024");
    expect(u.searchParams.get("nologo")).toBe("true");
  });

  it("respects a known model id", () => {
    const u = new URL(buildPollinationsUrl({ prompt: "x", model: "turbo" }));
    expect(u.searchParams.get("model")).toBe("turbo");
  });

  it("falls back to flux on an unknown model id", () => {
    const u = new URL(buildPollinationsUrl({ prompt: "x", model: "bogus-model" }));
    expect(u.searchParams.get("model")).toBe("flux");
  });

  it("clamps non-positive or non-numeric width/height to defaults", () => {
    const u1 = new URL(buildPollinationsUrl({ prompt: "x", width: 0, height: -10 }));
    expect(u1.searchParams.get("width")).toBe("1024");
    expect(u1.searchParams.get("height")).toBe("1024");
    const u2 = new URL(buildPollinationsUrl({ prompt: "x", width: 768, height: 512 }));
    expect(u2.searchParams.get("width")).toBe("768");
    expect(u2.searchParams.get("height")).toBe("512");
  });

  it("can disable nologo if explicitly requested", () => {
    const u = new URL(buildPollinationsUrl({ prompt: "x", noLogo: false }));
    expect(u.searchParams.get("nologo")).toBeNull();
  });

  it("throws on empty / whitespace-only prompt instead of silently building a broken URL", () => {
    expect(() => buildPollinationsUrl({ prompt: "" })).toThrow();
    expect(() => buildPollinationsUrl({ prompt: "   \t\n " })).toThrow();
  });
});

describe("cloudImage — model registry", () => {
  it("exposes the same model set used by the Settings select", () => {
    expect(IMAGE_MODEL_OPTIONS.map((o) => o.id)).toEqual(["flux", "turbo", "sdxl"]);
  });

  it("normalizePollinationsModel accepts known ids and rejects others", () => {
    expect(normalizePollinationsModel("flux")).toBe("flux");
    expect(normalizePollinationsModel("turbo")).toBe("turbo");
    expect(normalizePollinationsModel("sdxl")).toBe("sdxl");
    expect(normalizePollinationsModel(undefined)).toBe("flux");
    expect(normalizePollinationsModel("nope")).toBe("flux");
  });
});
