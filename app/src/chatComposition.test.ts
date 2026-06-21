import { describe, expect, it } from "vitest";
import {
  getIntentScanText,
  hasPastedTextPayload,
  isInlineTextTaskRequest,
  isTextCompositionRequest,
  isTextTransformRequest,
} from "./chatComposition";
import { isHolidayQuery, isLiveMediaQuery, needsWebSearch } from "./webSearch/intents";

const TONE_LETTER = `קרא את המכתב הבא ותגיד לי מה הטון של הכותב (כועס, מודאג, חגיגי) ואיך הגעת למסקנה הזו:"שלום רב, אני פונה אליכם לאחר שזו הפעם השלישית החודש שהמערכת שלכם קורסת."`;

const OPINION_ARTICLE = `הפוך את מאמר הדעה הזה לרשימת נקודות קצרות שמציגות בעד ונגד:"ההחלטה של עיריות רבות לאסור לחלוטין כניסת קורקינטים חשמליים שיתופיים למרכזי הערים מעוררת סערה. מצד אחד, מדובר בצעד קריטי להגנת הולכי הרגל; המדרכות הפכו לשדה קרב, ומספר התאונות בהן מעורבים קשישים וילדים זינק ב-40%. האיסור יחזיר את הביטחון לרחובות ויאלץ את הרוכבים להשתמש בתחבורה ציבורית. מנגד, המצדדים בכלי הרכב הללו מזכירים כי הקורקינט השיתופי הוא פתרון 'המייל האחרון' המושלם – הוא מפחית את הגודש בכבישים, מונע זיהום אוויר של מכוניות פרטיות, ומאפשר לצעירים ללא רישיון ניידות זולה ומהירה ברחבי העיר הפקוקה."`;

describe("chatComposition", () => {
  it("detects tone analysis of pasted complaint letter", () => {
    expect(hasPastedTextPayload(TONE_LETTER)).toBe(true);
    expect(isInlineTextTaskRequest(TONE_LETTER)).toBe(true);
    expect(needsWebSearch(TONE_LETTER)).toBe(false);
  });

  it("detects transform of pasted opinion article to pro/con list", () => {
    expect(isTextCompositionRequest(OPINION_ARTICLE)).toBe(true);
    expect(isTextTransformRequest(OPINION_ARTICLE)).toBe(true);
    expect(isInlineTextTaskRequest(OPINION_ARTICLE)).toBe(true);
    expect(needsWebSearch(OPINION_ARTICLE)).toBe(false);
    expect(isLiveMediaQuery(OPINION_ARTICLE)).toBe(false);
  });

  it("matches Hebrew transform verbs (JS \\b fails on Hebrew)", () => {
    expect(isTextCompositionRequest("הפוך את מאמר הדעה זה לרשימה")).toBe(true);
  });

  it("does not treat celebratory adjective as holiday query", () => {
    expect(isHolidayQuery("כועס, מודאג, חגיגי")).toBe(false);
    expect(isHolidayQuery("האם היום חג")).toBe(true);
  });

  it("still detects composition requests", () => {
    expect(isTextCompositionRequest("נסח את ההוראה הזו מחדש")).toBe(true);
  });

  it("does not block explicit live price queries", () => {
    expect(isInlineTextTaskRequest("כמה עולה חלב")).toBe(false);
    expect(needsWebSearch("כמה עולה חלב")).toBe(true);
  });

  it("intent scan uses instruction line for pasted payload without live intent", () => {
    const scan = getIntentScanText(OPINION_ARTICLE);
    expect(scan).toContain("הפוך");
    expect(scan).not.toContain("ילדים");
  });
});
