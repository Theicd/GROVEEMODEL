import { runWebSearch } from "../app/src/webSearch/orchestrator.ts";
import { extractCountryPhrase } from "../app/src/webSearch/queryExtract.ts";

const qs = [
  "מה השעה בטוקיו",
  "מה הבירה של גרמניה",
  "האם היום חג בגרמניה",
  "מי ראש הממשלה של ישראל",
  "מה מזג האוויר בתל אביב",
  "USD to ILS",
];

for (const q of qs) {
  console.log("\nQ:", q);
  console.log(" country:", extractCountryPhrase(q));
  const r = await runWebSearch(q);
  console.log(" intents:", r.intents.join(","));
  for (const s of r.sources) {
    console.log(`  ${s.provider} ok=${s.ok} ${s.error ?? s.text.slice(0, 100).replace(/\n/g, " ")}`);
  }
}
