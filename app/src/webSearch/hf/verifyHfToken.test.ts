import { describe, expect, it, vi, beforeEach } from "vitest";

vi.mock("../proxyFetch", () => ({
  proxyAwareFetch: vi.fn(),
}));

import { proxyAwareFetch } from "../proxyFetch";
import { verifyHfToken } from "./verifyHfToken";

describe("verifyHfToken", () => {
  beforeEach(() => {
    vi.mocked(proxyAwareFetch).mockReset();
  });

  it("rejects empty token", async () => {
    const r = await verifyHfToken("");
    expect(r.ok).toBe(false);
  });

  it("accepts valid whoami response", async () => {
    vi.mocked(proxyAwareFetch).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ name: "testuser" }),
    } as Response);
    const r = await verifyHfToken("hf_testtoken123");
    expect(r).toEqual({ ok: true, username: "testuser" });
  });

  it("rejects 401", async () => {
    vi.mocked(proxyAwareFetch).mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({}),
    } as Response);
    const r = await verifyHfToken("hf_bad");
    expect(r.ok).toBe(false);
  });
});
