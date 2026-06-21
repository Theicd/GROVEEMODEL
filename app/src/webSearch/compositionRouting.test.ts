import { describe, expect, it } from "vitest";
import { isInlineTextTaskRequest } from "../chatComposition";
import {
  classifySearchIntents,
  isProductsQuery,
  needsWebSearch,
} from "./intents";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import { isGameSearchRequest, shouldOpenGamePanel } from "../gameSearch/gameIntents";

const REPHRASE_GAME_BRIEF = `נסח את ההוראה הזו מחדש

שמתאים לחברת פיתוח בנושא לחברת תוכנה שמפתחת ממשקים וסטראטאפים ומגוון תחומים בהיי טק.

זה הניסוח

Design a high-end brand official משחק ארקייד בקובץ  HTML יחיד בסגנון שנות ה80 של משחקי הווידיאו עם גראפיקה משופרת עם שני או שלושה לחצים שיהיה פשוט מגניב תלתמ ימדי עם כל הנדרש ראליסטי ככל שניתן עם סופר ביצעים מותאם לטלפונים ומחשים כולל תפקיט וסאונד וצרת ניקוד וכוח וכל הנדקש והמתאים למשחקים מהתקופה

Visual Strategy:

Imagery: athletic poses, sweat, muscle definition, equipment texture.

Photography: high-contrast black and white, emphasizing strength and beauty.

Composition: people as the main subject, blurred surroundings.

Color Palette:

Primary Colors: pure black, dark gray.

Accent Colors: neon orange, metallic silver.

Background: dark, strong textural feel.

Typography:

Headings: bold sans-serif, powerful.

Body Text: clean and forceful, condensed information.

minimal whitespace, high information density.

Page Structure:

Hero Section: brand attitude + coach team showcase.

Curriculum: organized by goals(muscle gain, fat loss, fitness).

Coach Profiles: professional background + certifications.

Member Stories: real transformation case studies.

Book a Trial Class: clear CTA button.

Overall Vibe: professional, hardcore, motivating, sense of belonging.`;

describe("composition routing guard", () => {
  it("skips live search and canned replies for rephrase + pasted game/fitness brief", () => {
    expect(isInlineTextTaskRequest(REPHRASE_GAME_BRIEF)).toBe(true);
    expect(needsWebSearch(REPHRASE_GAME_BRIEF)).toBe(false);
    expect(isProductsQuery(REPHRASE_GAME_BRIEF)).toBe(false);
    expect(classifySearchIntents(REPHRASE_GAME_BRIEF)).toEqual([]);
    expect(isGameSearchRequest(REPHRASE_GAME_BRIEF)).toBe(false);
    expect(shouldOpenGamePanel(REPHRASE_GAME_BRIEF, "general")).toBe(false);
    expect(buildCapabilityLiveReply(REPHRASE_GAME_BRIEF, [], [])).toBeNull();
    expect(
      shouldDeliverStructuredLiveReply(
        REPHRASE_GAME_BRIEF,
        ["products"],
        [
          {
            provider: "israeli-products",
            label: "מוצרי סופר · ישראל",
            ok: true,
            text: "מסטיק אורביט",
            productHits: [],
          },
        ],
        "מסטיק אורביט",
      ),
    ).toBe(false);
  });

  it("does not treat super-performance Hebrew as supermarket query", () => {
    expect(isProductsQuery("משחק HTML עם סופר ביצועים לטלפונים")).toBe(false);
  });

  it("still detects explicit supermarket product queries", () => {
    expect(isProductsQuery("כמה עולה חלב")).toBe(true);
    expect(isProductsQuery("מחיר בסופר ללחם")).toBe(true);
    expect(needsWebSearch("כמה עולה חלב")).toBe(true);
  });
});
