import { describe, expect, it } from "vitest";
import {
  buildGradioData,
  parseGradioSseData,
  pickGradioEndpoint,
  resultLooksLikeImage,
  spaceIdToHost,
} from "./gradioSpaceClient";

describe("gradioSpaceClient", () => {
  it("maps space id to hf.space host", () => {
    expect(spaceIdToHost("black-forest-labs/FLUX.1-schnell")).toBe(
      "black-forest-labs-flux.1-schnell.hf.space",
    );
  });

  it("builds data with prompt in first string slot", () => {
    const data = buildGradioData(
      [
        { type: { type: "string" }, parameter_default: null },
        { type: { type: "number" }, parameter_default: 0 },
      ],
      "hello",
    );
    expect(data[0]).toBe("hello");
    expect(data[1]).toBe(0);
  });

  it("parses gradio SSE complete payload", () => {
    const text = `event: complete\ndata: [{"url":"https://x.hf.space/a.webp"},1]\n\n`;
    const parsed = parseGradioSseData(text);
    expect(parsed?.length).toBe(2);
    expect(resultLooksLikeImage(parsed!)).toBe(true);
  });

  it("picks infer endpoint from info", () => {
    const picked = pickGradioEndpoint(
      {
        named_endpoints: {
          "/infer": {
            parameters: [{ type: { type: "string" }, parameter_name: "prompt" }],
          },
          "/other": { parameters: [] },
        },
      },
      true,
    );
    expect(picked?.endpoint).toBe("/infer");
  });
});
