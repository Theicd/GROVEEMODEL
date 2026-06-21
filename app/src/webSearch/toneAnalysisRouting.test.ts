import { describe, expect, it } from "vitest";
import { isInlineTextTaskRequest } from "../chatComposition";
import { classifySearchIntents, needsWebSearch } from "./intents";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";

const TONE_LETTER = `קרא את המכתב הבא ותגיד לי מה הטון של הכותב (כועס, מודאג, חגיגי) ואיך הגעת למסקנה הזו:"שלום רב, אני פונה אליכם לאחר שזו הפעם השלישית החודש שהמערכת שלכם קורסת במהלך שעות הפעילות המרכזיות של העסק שלי. שלחתי כבר שני מיילים לתמיכה הטכנית וכל מה שקיבלתי זו תשובה אוטומטית ש'הנושא בטיפול'. המצב הזה פוגע ישירות בהכנסות שלי וגורם לי לאבד לקוחות יקרים בכל יום שעובר. אני באמת מעריך את המוצר שלכם ועובד איתו שנים, אבל אם לא נקבל פתרון קונקרטי ומיידי ב-24 השעות הקרובות, פשוט לא תהיה לי ברירה אלא לבטל את המנוי השנתי ולעבור למתחרים."`;

describe("tone letter routing", () => {
  it("skips web search for pasted letter tone analysis", () => {
    expect(isInlineTextTaskRequest(TONE_LETTER)).toBe(true);
    expect(needsWebSearch(TONE_LETTER)).toBe(false);
    expect(classifySearchIntents(TONE_LETTER)).toEqual([]);
    expect(buildCapabilityLiveReply(TONE_LETTER, [], [])).toBeNull();
    expect(
      shouldDeliverStructuredLiveReply(
        TONE_LETTER,
        ["products"],
        [{ provider: "israeli-products", label: "x", ok: true, text: "x", productHits: [] }],
        "x",
      ),
    ).toBe(false);
  });
});
