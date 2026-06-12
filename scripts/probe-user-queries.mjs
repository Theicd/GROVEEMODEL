/** Probe the exact user-reported failing queries. */
import { runWebSearch } from "../app/src/webSearch/orchestrator.ts";
import { classifySearchIntents } from "../app/src/webSearch/intents.ts";
import { extractCurrencyPair, extractPlacePair } from "../app/src/webSearch/queryExtract.ts";

const qs = [
  "מה המטבע של ברזיל?",
  "מה היחס שלו לדולר כמה BRL אני קונה ב1 דולר",
  "מצא בית חולים ליד מגדל אייפel",
  "אילו תחנות רכbet יש ליד שדה התעופה הית'רo?",
  'כמה ק"מ בין ירושלים לחיפה?',
  "מה הכותרת הראשית באתר BBC עכשיו?",
  "מהם 10 הנושאים המסוקרים ביותר בעולם כרגע?",
];

for (const q of qs) {
  console.log("\n" + "=".repeat(60));
  console.log("Q:", q);
  console.log(" intents:", classifySearchIntents(q).join(", "));
  const r = await runWebSearch(q);
  for (const s of r.sources) {
    const preview = s.ok ? s.text.split("\n").slice(0, 4).join(" | ") : s.error;
    console.log(`  ${s.ok ? "✓" : "✗"} ${s.provider}: ${preview}`);
  }
}
