import { describe, expect, it, beforeEach } from "vitest";
import {
  deleteCustomQuery,
  hideBuiltinQuery,
  loadEffectiveQueries,
  nextCustomQueryId,
  resetQaQueryOverrides,
  upsertCustomQuery,
} from "./presentationQaQueryStore";
import { USER_PRESENTATION_QUERIES } from "./userPresentationQueries";

const store = new Map<string, string>();

describe("presentationQaQueryStore", () => {
  beforeEach(() => {
    store.clear();
    Object.defineProperty(globalThis, "localStorage", {
      value: {
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => store.set(k, v),
        removeItem: (k: string) => store.delete(k),
      },
      configurable: true,
    });
    resetQaQueryOverrides();
  });

  it("loads builtin queries by default", () => {
    const q = loadEffectiveQueries();
    expect(q.length).toBeGreaterThanOrEqual(USER_PRESENTATION_QUERIES.length);
    expect(q.some((x) => x.id === "B01")).toBe(true);
  });

  it("hides builtin and adds custom", () => {
    hideBuiltinQuery("B01");
    upsertCustomQuery({
      id: "CUST-001",
      group: "basic",
      category: "test",
      prompt: "שאלת בדיקה",
      custom: true,
    });
    const q = loadEffectiveQueries();
    expect(q.some((x) => x.id === "B01")).toBe(false);
    expect(q.some((x) => x.id === "CUST-001")).toBe(true);
  });

  it("generates next custom id", () => {
    expect(nextCustomQueryId([])).toBe("CUST-001");
    upsertCustomQuery({
      id: "CUST-001",
      group: "basic",
      category: "x",
      prompt: "y",
      custom: true,
    });
    expect(nextCustomQueryId(loadEffectiveQueries())).toBe("CUST-002");
  });

  it("deletes custom query", () => {
    upsertCustomQuery({
      id: "CUST-001",
      group: "basic",
      category: "x",
      prompt: "y",
      custom: true,
    });
    deleteCustomQuery("CUST-001");
    expect(loadEffectiveQueries().some((x) => x.id === "CUST-001")).toBe(false);
  });
});
