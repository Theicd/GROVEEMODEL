/** Batch QA for live web-search providers (no LLM). */
import { runWebSearch } from "../app/src/webSearch/orchestrator.ts";
import { needsWebSearch } from "../app/src/webSearch/intents.ts";

const QUERIES = [
  { q: "מה השעה עכשיו בניו יורק", tag: "שעון" },
  { q: "what time is it in London", tag: "שעון EN" },
  { q: "מה התאריך היום", tag: "תאריך (ללא חיפוש)" },
  { q: "מה מזג האוויר בתל אביב", tag: "מזג אוויר" },
  { q: "weather in Miami today", tag: "מזג EN" },
  { q: "גובה גלים בתל אביב", tag: "ים" },
  { q: "רעידות אדמה אחרונות", tag: "רעידות" },
  { q: "מי ראש הממשלה של ישראל", tag: "ממשל" },
  { q: "מי נשיא ארה\"ב", tag: "ממשל US" },
  { q: "מה הבירה של צרפת", tag: "מדינות" },
  { q: "האם היום חג בגרמניה", tag: "חגים" },
  { q: "USD to ILS exchange rate", tag: "מטבע" },
  { q: "היי מה שלומך", tag: "צ'אט (ללא חיפוש)" },
  { q: "חפש מידע על פירמידות", tag: "ויקיפedia" },
  { q: "מה קורה עם React hooks", tag: "GitHub/Wiki" },
];

let pass = 0;
let fail = 0;

for (const { q, tag } of QUERIES) {
  const shouldSearch = needsWebSearch(q);
  const r = await runWebSearch(q);
  const okSources = r.sources.filter((s) => s.ok);
  const failed = r.sources.filter((s) => !s.ok);

  const isChatOnly = !shouldSearch;
  const success = isChatOnly
    ? r.sources.length === 0
    : okSources.length > 0;

  if (success) pass++;
  else fail++;

  const status = success ? "✅" : "❌";
  console.log(`\n${status} [${tag}] ${q}`);
  console.log(`   search=${shouldSearch} intents=${r.intents.join(",") || "—"}`);
  if (okSources.length) {
    for (const s of okSources) {
      console.log(`   ✓ ${s.provider}: ${s.text.split("\n")[0].slice(0, 90)}`);
    }
  }
  if (failed.length) {
    for (const s of failed) {
      console.log(`   ✗ ${s.provider}: ${s.error}`);
    }
  }
  if (isChatOnly && r.sources.length) {
    console.log(`   ⚠ unexpected search for casual chat`);
  }
}

console.log(`\n--- סיכום: ${pass}/${QUERIES.length} עברו ---`);
process.exit(fail > 0 ? 1 : 0);
