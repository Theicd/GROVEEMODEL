import { describe, expect, it } from "vitest";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import { classifySearchIntents, isNewsQuery } from "./intents";
import { regexPlanForQuery } from "./searchPlanner";
import type { SearchSourceResult } from "./types";

const B03 = "מה שער הדולר מול השקל כרגע?";

describe("B03 currency QA path", () => {
  it("routes to currency only — not news", () => {
    const intents = classifySearchIntents(B03);
    expect(intents).toContain("currency");
    expect(isNewsQuery(B03)).toBe(false);
    expect(intents.includes("news")).toBe(false);
  });

  it("isNewsQuery stays false for B03 prompt", async () => {
    const { isNewsQuery: newsQ, needsWebSearch } = await import("./intents");
    expect(newsQ(B03)).toBe(false);
    expect(needsWebSearch(B03)).toBe(true);
  });

  it("regex plan uses short_fact", () => {
    const plan = regexPlanForQuery(B03);
    expect(plan?.intents).toContain("currency");
    expect(plan?.answerShape).toBe("short_fact");
  });

  it("builds canned reply with USD/ILS rate", () => {
    const intents = classifySearchIntents(B03);
    const src: SearchSourceResult = {
      provider: "frankfurter-fx",
      label: "שערי מטבע (Frankfurter)",
      ok: true,
      text: "תאריך: 2026-06-19\n1 USD = 2.9622 ILS\n100 USD = 296.2200 ILS\nמקור: European Central Bank via Frankfurter",
      latencyMs: 12,
    };
    const reply = buildCapabilityLiveReply(B03, intents, [src]);
    expect(reply).toMatch(/2\.9622/);
    expect(reply).toMatch(/USD =/);
    expect(shouldDeliverStructuredLiveReply(B03, intents, [src], reply)).toBe(true);
  });

  it("shouldDeliverStructuredLiveReply stays true when news intent leaked", () => {
    const intents = ["currency", "news"] as const;
    const src: SearchSourceResult = {
      provider: "frankfurter-fx",
      label: "Frankfurter",
      ok: true,
      text: "תאריך: 2026-06-19\n1 USD = 2.9622 ILS",
      latencyMs: 1,
    };
    const reply = buildCapabilityLiveReply(B03, [...intents], [src]);
    expect(shouldDeliverStructuredLiveReply(B03, [...intents], [src], reply)).toBe(true);
  });

  it("builds dual USD+EUR→ILS composite reply", () => {
    const q = "מה שער הדולר והיורו מול השקל";
    const intents = classifySearchIntents(q);
    const src: SearchSourceResult = {
      provider: "frankfurter-fx",
      label: "שערי מטבע (Frankfurter)",
      ok: true,
      text: [
        "תאריך: 2026-06-19",
        "1 USD = 3.70 ILS",
        "1 EUR = 4.02 ILS",
        "מקור: European Central Bank via Frankfurter",
      ].join("\n"),
      latencyMs: 8,
    };
    const reply = buildCapabilityLiveReply(q, intents, [src]);
    expect(reply).toMatch(/USD = 3\.70/);
    expect(reply).toMatch(/EUR = 4\.02/);
  });
});
