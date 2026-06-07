import { describe, expect, it } from "vitest";
import { buildPersistedAssistantPayload, extractPrimaryArtifact } from "./artifacts";

const SAMPLE_HTML = `\`\`\`html
<!DOCTYPE html>
<html><head><style>
.star { position: absolute; }
</style></head>
<body><div class="star"></div>
<script>
const stars = [];
for (let i = 0; i < 50; i++) stars.push(i);
</script></body></html>
\`\`\``;

describe("artifacts persistence", () => {
  it("preserves HTML verbatim in artifact field", () => {
    const raw = `<|channel>thought
plan animation

${SAMPLE_HTML}`;
    const { content, artifact } = buildPersistedAssistantPayload(raw, true);
    expect(artifact?.kind).toBe("html");
    expect(artifact?.content).toContain("class=\"star\"");
    expect(artifact?.content).toContain("for (let i = 0; i < 50; i++)");
    expect(content).not.toContain("<script>");
  });

  it("extractPrimaryArtifact matches streaming and saved forms", () => {
    const a = extractPrimaryArtifact(SAMPLE_HTML);
    expect(a?.content).toContain("<!DOCTYPE html>");
  });
});
