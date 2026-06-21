import { describe, expect, it } from "vitest";
import { isInlineTextTaskRequest } from "../chatComposition";
import { classifySearchIntents, isLiveMediaQuery, needsWebSearch } from "./intents";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";

const OPINION_ARTICLE = `הפוך את מאמר הדעה הזה לרשימת נקודות קצרות שמציגות בעד ונגד:"ההחלטה של עיריות רבות לאסור לחלוטין כניסת קורקינטים חשמליים שיתופיים למרכזי הערים מעוררת סערה. מצד אחד, מדובר בצעד קריטי להגנת הולכי הרגל; המדרכות הפכו לשדה קרב, ומספר התאונות בהן מעורבים קשישים וילדים זינק ב-40%. האיסור יחזיר את הביטחון לרחובות ויאלץ את הרוכבים להשתמש בתחבורה ציבורית. מנגד, המצדדים בכלי הרכב הללו מזכירים כי הקורקינט השיתופי הוא פתרון 'המייל האחרון' המושלם – הוא מפחית את הגודש בכבישים, מונע זיהום אוויר של מכוניות פרטיות, ומאפשר לצעירים ללא רישיון ניידות זולה ומהירה ברחבי העיר הפקוקה."`;

describe("opinion article routing", () => {
  it("skips TV/live search for pro/con list transform", () => {
    expect(isInlineTextTaskRequest(OPINION_ARTICLE)).toBe(true);
    expect(needsWebSearch(OPINION_ARTICLE)).toBe(false);
    expect(isLiveMediaQuery(OPINION_ARTICLE)).toBe(false);
    expect(classifySearchIntents(OPINION_ARTICLE)).toEqual([]);
    expect(buildCapabilityLiveReply(OPINION_ARTICLE, [], [])).toBeNull();
    expect(
      shouldDeliverStructuredLiveReply(
        OPINION_ARTICLE,
        ["livemedia"],
        [{ provider: "iptv", label: "TV LIVE / Radio", ok: true, text: "Kids channel", productHits: [] }],
        "Kids",
      ),
    ).toBe(false);
  });

  it("still searches live TV when explicitly requested", () => {
    expect(needsWebSearch("תן לי ערוץ ילדים בטלוויזיה")).toBe(true);
    expect(isLiveMediaQuery("ערוץ ילדים")).toBe(true);
  });
});
