import { describe, expect, it } from "vitest";
import { wantsWorksheetReplicaHtml } from "./chatIntents";
import {
  buildSimpleWorksheetFromPlainText,
  buildWorksheetReplicaHtml,
  enhanceWorksheetReplicaHtml,
  isWorksheetReplicaArtifact,
} from "./worksheetReplicaHtml";

describe("worksheetReplicaHtml", () => {
  it("detects worksheet replica intent", () => {
    expect(wantsWorksheetReplicaHtml("חלץ את השאלות מהדף וצור קובץ HTML זהה לתמונה")).toBe(true);
    expect(wantsWorksheetReplicaHtml("מה כתוב בתמונה")).toBe(false);
  });

  it("builds A4 fillable page with print styles", () => {
    const html = buildWorksheetReplicaHtml({
      title: "מתמטיקה",
      body: '<div class="ws-q"><span class="ws-q-num">1.</span><span class="ws-q-text">2+2=</span></div><input class="ws-answer" type="text" />',
    });
    expect(html).toContain("worksheet-replica");
    expect(html).toContain("@media print");
    expect(html).toContain('type="text"');
  });

  it("builds numbered inputs from plain OCR text", () => {
    const html = buildSimpleWorksheetFromPlainText("1. מהי בירת ישראל?\n2. כמה זה 3×4?", "בוחן");
    expect(isWorksheetReplicaArtifact(html)).toBe(true);
    expect(html).toContain("ws-answer");
    expect(html).toContain("בירת ישראל");
  });

  it("wraps weak HTML in worksheet shell", () => {
    const weak = "<p>שאלה 1: _____</p>";
    const out = enhanceWorksheetReplicaHtml(weak, "דף עבודה");
    expect(out).toContain("worksheet-replica");
    expect(out).toContain("שאלה 1");
  });
});
