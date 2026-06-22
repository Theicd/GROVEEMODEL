import { describe, expect, it } from "vitest";
import { buildPanelSearchPlan } from "./panelSearch";

describe("buildPanelSearchPlan", () => {
  it("uses focused cinema query instead of full chat sentence", () => {
    const q =
      "חפש באינטרנט: מהם 3 הסרטים הכי מצליחים שמציגים עכשיו בבתי הקולנוע בישראל? תן תקציר";
    const plan = buildPanelSearchPlan(q);
    expect(plan.queries[0]).toMatch(/קולנוע|hotcinema|seret|box office/i);
    expect(plan.queries[0]).not.toMatch(/תקציר|מהם 3/);
    expect(plan.blendNewsWithWeb).toBe(false);
  });

  it("uses focused Euro query for sports championship", () => {
    const q = "מי זכתה באליפות היורו ומי שחקן המצטיין";
    const plan = buildPanelSearchPlan(q);
    expect(plan.queries[0]).toMatch(/UEFA|Euro|player/i);
    expect(plan.useWebFallback).toBe(true);
  });
});
