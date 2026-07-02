import { describe, expect, it } from "vitest";
import {
  hasUnclosedCodeFence,
  isCameraContextQuestion,
  isContinueRequest,
  isRtlText,
  isSimpleGreeting,
  isPersonActivityQuestion,
  isPersonVisibilityQuestion,
  isCurrentPersonStateQuestion,
  isFingerCountQuestion,
  needsPersonFocusRefresh,
  isSceneInterpretationQuestion,
  isVisualDetailQuestion,
  needsCameraVisionEscalation,
  parseGemmaThinkingOutput,
  getArtifactScanContent,
  splitAssistantStream,
  shouldContinueCode,
  stripGemmaControlTokens,
  trimHistoryForContext,
  classifyChatTopic,
  isTopicShift,
  topicShiftHint,
  isConversationFirstRequest,
  isConversationalChatTurn,
  isHypotheticalOrCreativeQuery,
  isPersonalOrEmotionalSharing,
  formatCameraTopicLabel,
  needsVisionSensorContext,
  isVisionUnrelatedTurn,
  needsAttachedDocumentAnalysis,
  isDocumentImageRequest,
  wantsWorksheetReplicaHtml,
  wantsExactTextExtraction,
  stripLeakedVisionReport,
} from "./chatIntents";

describe("chatIntents", () => {
  it("RTL heuristic", () => {
    expect(isRtlText("שלום")).toBe(true);
    expect(isRtlText("hello")).toBe(false);
  });

  it("simple greetings", () => {
    expect(isSimpleGreeting("היי")).toBe(true);
    expect(isSimpleGreeting("hello")).toBe(true);
    expect(isSimpleGreeting("שלום גרובי")).toBe(true);
    expect(isSimpleGreeting("hi groovie")).toBe(true);
    expect(isSimpleGreeting("צור תמונה")).toBe(false);
  });

  it("camera context questions", () => {
    expect(isCameraContextQuestion("מה אתה רואה?")).toBe(true);
    expect(isCameraContextQuestion("what do you see")).toBe(true);
    expect(isCameraContextQuestion("כתוב html")).toBe(false);
  });

  it("visual detail questions trigger vision escalation", () => {
    expect(isVisualDetailQuestion("אתה רואה מה השעה בשעון?")).toBe(true);
    expect(isVisualDetailQuestion("מה כתוב על המסך?")).toBe(true);
    expect(isVisualDetailQuestion("איזה צבע החולצה?")).toBe(true);
    expect(isVisualDetailQuestion("what time is on the clock")).toBe(true);
    expect(isVisualDetailQuestion("שלום")).toBe(false);
    expect(needsCameraVisionEscalation("מה השעה בשעון?")).toBe(true);
    expect(needsCameraVisionEscalation("מה אתה רואה?")).toBe(true);
    expect(isPersonVisibilityQuestion("אתה רואה אותי?")).toBe(true);
    expect(needsCameraVisionEscalation("אתה רואה אותי?")).toBe(true);
    expect(isPersonActivityQuestion("מה האדם עושה עכשיו")).toBe(true);
    expect(isCurrentPersonStateQuestion("האדם עומד או יושב?")).toBe(true);
    expect(needsPersonFocusRefresh("האדם עומד או יושב?")).toBe(true);
    expect(needsCameraVisionEscalation("האדם עומד או יושב?")).toBe(true);
    expect(isSceneInterpretationQuestion("מה אתה רואה?")).toBe(true);
    expect(isSceneInterpretationQuestion("מה השעה בשעון?")).toBe(false);
    expect(isFingerCountQuestion("כמה אצבעות אתה רואה?")).toBe(true);
    expect(isFingerCountQuestion("how many fingers do you see")).toBe(true);
    expect(needsCameraVisionEscalation("כמה אצבעות?")).toBe(true);
    expect(needsPersonFocusRefresh("כמה אצבעות?")).toBe(true);
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

  it("classifies chat topics", () => {
    expect(classifyChatTopic("משהו מינימליסטי יהיה הכיוון")).toBe("design");
    expect(classifyChatTopic("משעמם לי מה אתה מציע שנשחק?")).toBe("bored_play");
    expect(classifyChatTopic("מה אתה רואה?")).toBe("camera");
  });

  it("detects topic shift design to bored_play", () => {
    expect(isTopicShift("design", "bored_play")).toBe(true);
    expect(isTopicShift("design", "design")).toBe(false);
    expect(topicShiftHint("design", "bored_play")).toMatch(/Do NOT continue/);
  });

  it("conversation-first requests include play and sharing", () => {
    expect(isConversationFirstRequest("בוא נשחק משחק")).toBe(true);
    expect(isConversationFirstRequest("הפסקת חשמל")).toBe(true);
    expect(isConversationFirstRequest("מה אתה רואה?")).toBe(false);
  });

  it("formats camera topic labels for UI", () => {
    expect(formatCameraTopicLabel("bored_play")).toBe("משחק / שעמום");
    expect(formatCameraTopicLabel("pack:acknowledgment-request")).toBe("רגע / מצב");
  });

  it("splits vision vs pure chat turns", () => {
    expect(needsVisionSensorContext("מה אתה רואה סביבך")).toBe(true);
    expect(needsVisionSensorContext("אני מחפש רעיון לסיפור מדע בדיוני")).toBe(false);
    expect(needsVisionSensorContext("שלום איך קוראים לך")).toBe(false);
    expect(needsVisionSensorContext("תגיד לי אתה תן לי רעיון")).toBe(false);
    expect(isVisionUnrelatedTurn("מה אני מחזיק ביד")).toBe(false);
  });

  it("conversation-first and hypothetical turns skip live lookup routing", () => {
    expect(isHypotheticalOrCreativeQuery("נניח שאנשים יכולים לעוף")).toBe(true);
    expect(isConversationalChatTurn("מה דעתך על מדע בדיוני?")).toBe(true);
    expect(isConversationalChatTurn("בוא נדבר על רעיון לסיפור")).toBe(true);
    expect(isConversationalChatTurn("what do you think about AI ethics?")).toBe(true);
    expect(isPersonalOrEmotionalSharing("איחרתי לעבודה היום הבוס כועס עלי")).toBe(true);
    expect(isConversationalChatTurn("איחרתי לעבודה היום הבוס כועס עלי")).toBe(true);
    expect(isConversationalChatTurn("מה מזג האוויר בתל אביב")).toBe(false);
    expect(isConversationalChatTurn("חפש מידע על ברמודה")).toBe(false);
  });

  it("routes attached images as document analysis", () => {
    expect(needsAttachedDocumentAnalysis("שלום", false)).toBe(false);
    expect(needsAttachedDocumentAnalysis("שלום", true)).toBe(true);
    expect(isDocumentImageRequest("תסתכל על התמונה אלו השאלות")).toBe(true);
    expect(isDocumentImageRequest("אני צריך עזרה בשיעורי הבית")).toBe(true);
    expect(isDocumentImageRequest("")).toBe(true);
    expect(isDocumentImageRequest("שלום")).toBe(false);
  });

  it("detects worksheet HTML replica requests", () => {
    expect(wantsWorksheetReplicaHtml("חלץ את השאלות וצור HTML זהה לתמונה")).toBe(true);
    expect(wantsWorksheetReplicaHtml("פתור את השאלות")).toBe(false);
  });

  it("runs OCR only for exact transcription requests", () => {
    expect(wantsExactTextExtraction("מה כתוב בתמונה")).toBe(true);
    expect(wantsExactTextExtraction("חלץ שאלות ועזור לי לפתור")).toBe(false);
  });

  it("strips leaked perception snapshot from output", () => {
    const raw = `Perception snapshot (internal — not for user display):
Person visible: NO
YOLO persons: 0

שלום, איך אוכל לעזור?`;
    expect(stripLeakedVisionReport(raw)).toBe("שלום, איך אוכל לעזור?");
  });
});
