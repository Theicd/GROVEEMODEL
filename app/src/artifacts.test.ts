import { describe, expect, it } from "vitest";
import { buildPersistedAssistantPayload } from "./artifacts";
import { stripHtmlFencesForChat } from "./chatMarkdown";

describe("chat-only document replies", () => {
  it("flattens HTML fences into chat markdown text", () => {
    const raw = `הנה התוכן:

\`\`\`html
<!DOCTYPE html><html><body><h1>שאלה 1</h1><p>2+2=</p></body></html>
\`\`\``;
    const flat = stripHtmlFencesForChat(raw);
    expect(flat).toContain("שאלה 1");
    expect(flat).not.toContain("<!DOCTYPE");
  });

  it("does not create artifact for document chat-only mode", () => {
    const raw = `\`\`\`html
<html><body><p>1. מהי בירת ישראל?</p></body></html>
\`\`\``;
    const { content, artifact } = buildPersistedAssistantPayload(raw, false, {
      chatOnlyDocument: true,
    });
    expect(artifact).toBeNull();
    expect(content).toContain("בירת ישראל");
  });
});
