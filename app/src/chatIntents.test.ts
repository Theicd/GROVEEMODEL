import { describe, expect, it } from "vitest";
import {
  hasUnclosedCodeFence,
  isContinueRequest,
  isRtlText,
  isSimpleGreeting,
  parseGemmaThinkingOutput,
  getArtifactScanContent,
  splitAssistantStream,
  shouldContinueCode,
  stripGemmaControlTokens,
  trimHistoryForContext,
} from "./chatIntents";

describe("chatIntents", () => {
  it("RTL heuristic", () => {
    expect(isRtlText("שלום")).toBe(true);
    expect(isRtlText("hello")).toBe(false);
  });

  it("simple greetings", () => {
    expect(isSimpleGreeting("היי")).toBe(true);
    expect(isSimpleGreeting("hello")).toBe(true);
    expect(isSimpleGreeting("צור תמונה")).toBe(false);
  });

  it("continue requests", () => {
    expect(isContinueRequest("המשך לכתוב את הקוד")).toBe(true);
    expect(isContinueRequest("continue writing")).toBe(true);
    expect(isContinueRequest("מה השעה")).toBe(false);
  });

  it("unclosed code fence", () => {
    expect(hasUnclosedCodeFence("```html\n<div>")).toBe(true);
    expect(hasUnclosedCodeFence("```html\n<div>\n```")).toBe(false);
  });

  it("shouldContinueCode", () => {
    const turns = [
      { role: "user" as const, content: "כתוב html" },
      { role: "assistant" as const, content: "```html\n<canvas" },
    ];
    expect(shouldContinueCode("המשך לכתוב", turns)).toBe(true);
    expect(shouldContinueCode("תודה", turns)).toBe(false);
  });

  it("trimHistoryForContext pins last assistant", () => {
    const turns = [
      { role: "user" as const, content: "a".repeat(1000) },
      { role: "assistant" as const, content: "b".repeat(5000) },
      { role: "user" as const, content: "continue" },
    ];
    const trimmed = trimHistoryForContext(turns, 6000, true);
    expect(trimmed.some((t) => t.role === "assistant" && t.content.length === 5000)).toBe(true);
  });

  it("parseGemmaThinkingOutput splits thought from answer", () => {
    const raw = "<|channel>thought\nstep one\n\nFinal answer here.";
    const p = parseGemmaThinkingOutput(raw);
    expect(p.hasThinking).toBe(true);
    expect(p.thought).toContain("step one");
    expect(p.answer).toContain("Final answer");
  });

  it("stripGemmaControlTokens keeps answer only", () => {
    const raw = "<|channel>thought\nhidden\n\nHello world";
    expect(stripGemmaControlTokens(raw)).toBe("Hello world");
  });

  it("splitAssistantStream ignores ```html mentions inside thought", () => {
    const raw = `thought
Thinking Process:
1. Use a \`\`\`html fence for output

\`\`\`html
<!DOCTYPE html><html><body>Hi</body></html>
\`\`\``;
    const parts = splitAssistantStream(raw, true);
    expect(parts.thinkingInProgress).toBe(false);
    expect(parts.answer).toMatch(/^```html/m);
    expect(parts.thought).toContain("Thinking Process");
    expect(getArtifactScanContent(raw, true)).toContain("<!DOCTYPE");
  });

  it("getArtifactScanContent empty while thinking in progress", () => {
    const raw = "thought\nStill planning, no code yet.";
    expect(getArtifactScanContent(raw, true)).toBe("");
  });
});
