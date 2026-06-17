import { describe, expect, it } from "vitest";
import { shouldRetainSearchFocus } from "./searchInteraction";

describe("shouldRetainSearchFocus", () => {
  it("returns true when related target is inside the form", () => {
    const child = { nodeType: 1 } as unknown as Node;
    const form = { contains: (n: Node) => n === child } as unknown as HTMLFormElement;
    expect(shouldRetainSearchFocus(child, form)).toBe(true);
  });

  it("returns false for unrelated targets", () => {
    const other = { nodeType: 1 } as unknown as Node;
    const form = { contains: () => false } as unknown as HTMLFormElement;
    expect(shouldRetainSearchFocus(other, form)).toBe(false);
  });
});
