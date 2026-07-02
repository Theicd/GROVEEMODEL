/**
 * Session-scoped user facts ("remember X") — regex only, no LLM.
 * SmolLM often loses mid-chat facts; we extract, pin, inject, and recall deterministically.
 */

import type { UiLang } from "./chatRoutePrelude";

const MEMORY_SAVE_RE =
  /^(?:זכור|תזכור|שמור(?:\s+ב(?:ז)?כרון)?|remember|don't\s+forget|dont\s+forget)\s*[:\-]?\s*(.+)/is;

const MEMORY_RECALL_RE =
  /(?:מה\s+(?:ה)?עיר\s+(?:ה)?(?:אהובה|מועדפת)\s+(?:עלי|שלי|on\s+me)|איז(?:ו|ה)\s+עיר\s+(?:אני\s+)?(?:אוהב|אהובה)|what\s+(?:is\s+)?my\s+favorite\s+city|what\s+city\s+do\s+i\s+love|what\s+did\s+i\s+(?:tell|ask)\s+you\s+to\s+remember|מה\s+(?:אמרתי|ביקשתי)\s+(?:ל)?זכור|מה\s+ש(?:מור|מרתי)|איפה\s+אמרתי\s+ש(?:אני\s+)?אוהב)/i;

export function isUserMemorySaveRequest(text: string): boolean {
  return MEMORY_SAVE_RE.test(text.trim());
}

export function extractMemoryFactFromSave(text: string): string | null {
  const m = text.trim().match(MEMORY_SAVE_RE);
  return m?.[1]?.trim().replace(/\s+/g, " ") || null;
}

export function isUserMemoryRecallQuery(text: string): boolean {
  const t = text.trim();
  if (!t || t.length > 160) return false;
  return MEMORY_RECALL_RE.test(t);
}

export function collectSessionMemoryFacts(
  messages: Array<{ role: string; content: string }>,
): string[] {
  const facts: string[] = [];
  for (const m of messages) {
    if (m.role !== "user") continue;
    const fact = extractMemoryFactFromSave(m.content);
    if (fact && !facts.includes(fact)) facts.push(fact);
  }
  return facts;
}

export function memoryPinnedSourceIndices(
  entries: Array<{ role: string; content: string }>,
): number[] {
  const pinned = new Set<number>();
  for (let i = 0; i < entries.length; i++) {
    if (entries[i].role !== "user" || !isUserMemorySaveRequest(entries[i].content)) continue;
    pinned.add(i);
    if (entries[i + 1]?.role === "assistant") pinned.add(i + 1);
  }
  return [...pinned].sort((a, b) => a - b);
}

export function extractCityFromMemoryFact(fact: string): string | null {
  const t = fact.trim();
  const patterns = [
    /(?:העיר\s+(?:ה)?(?:אהובה|מועדפת).*?(?:היא|הוא))\s+([^\n,.!?]+)/i,
    /(?:favorite\s+city\s+(?:is|on\s+me\s+is))\s+([^\n,.!?]+)/i,
    /(?:אוהב(?:ת|ים)?\s*(?:את|את\s+)?)\s*([^\n,.!?]+)/i,
    /(?:love\s+(?:the\s+city\s+of\s+)?)\s*([^\n,.!?]+)/i,
    /(?:היא|is)\s+([A-Za-z\u0590-\u05FF][^\n,.!?]*)/i,
  ];
  for (const re of patterns) {
    const m = t.match(re);
    const city = m?.[1]?.trim();
    if (city && city.length >= 2 && city.length <= 48) return city;
  }
  return null;
}

export function formatSessionMemoryForPrompt(facts: string[]): string {
  if (!facts.length) return "";
  const lines = facts.map((f) => `- ${f}`).join("\n");
  return `User facts from this chat (trust these over older chitchat):\n${lines}\nWhen asked about user preferences, use ONLY these facts.`;
}

export function answerSessionMemoryRecall(
  query: string,
  facts: string[],
  uiLang: UiLang,
): string | null {
  if (!facts.length) {
    if (!isUserMemoryRecallQuery(query)) return null;
    return uiLang === "he"
      ? "לא שמרתי עדיין עובדה כזו בשיחה. אפשר לכתוב «זכור: …» ואז אשתמש בזה."
      : "I don't have a saved fact like that yet. Say «remember: …» and I'll use it.";
  }

  if (!isUserMemoryRecallQuery(query)) return null;

  if (/עיר|city/i.test(query)) {
    for (let i = facts.length - 1; i >= 0; i--) {
      const city = extractCityFromMemoryFact(facts[i]);
      if (city) {
        return uiLang === "he"
          ? `אמרת שהעיר האהובה עליך היא **${city}**.`
          : `You said your favorite city is **${city}**.`;
      }
    }
  }

  const list = facts.map((f) => `• ${f}`).join("\n");
  return uiLang === "he" ? `שמרתי בשיחה:\n${list}` : `You asked me to remember:\n${list}`;
}
