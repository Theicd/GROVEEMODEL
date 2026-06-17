import { describe, expect, it } from "vitest";
import { decodeHtmlEntities } from "./decodeHtmlEntities";

describe("decodeHtmlEntities", () => {
  it("decodes numeric and named entities", () => {
    expect(decodeHtmlEntities("Xbox &#8211; Report")).toBe("Xbox – Report");
    expect(decodeHtmlEntities("Tom &amp; Jerry")).toBe("Tom & Jerry");
  });
});
