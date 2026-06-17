/**
 * Print NEWS chat QA checklist + validate routing for all 10 questions.
 * Live panel/cards: run dev server, paste each query in chat, tick verifyHe.
 */
import { classifySearchIntents } from "../app/src/webSearch/intents.ts";
import { needsWebSearch } from "../app/src/webSearch/intents.ts";
import { isTopicsOverviewQuery } from "../app/src/groveeNews/headlineIntent.ts";
import { normalizeNewsEngineQuery } from "../app/src/groveeNews/newsQueryNormalize.ts";
import { NEWS_ACCEPTANCE_QUERIES } from "../app/src/groveeNews/newsAcceptanceQueries.ts";

let failed = 0;

console.log("\n=== GROVEE NEWS — chat QA questions ===\n");

for (const q of NEWS_ACCEPTANCE_QUERIES) {
  const intents = classifySearchIntents(q.query);
  const topics = isTopicsOverviewQuery(q.query);
  const panel = topics ? "topics" : "search";
  const engineQ = panel === "search" ? normalizeNewsEngineQuery(q.query) : "(topics lanes)";
  const intentOk = q.expectIntents.every((i) => intents.includes(i));
  const panelOk = panel === q.expectPanelMode;
  const engineOk =
    !q.expectEngineQuery ||
    normalizeNewsEngineQuery(q.query).toLowerCase().includes(q.expectEngineQuery.toLowerCase());
  const webOk = needsWebSearch(q.query);
  const pass = intentOk && panelOk && engineOk && webOk;

  if (!pass) failed += 1;

  console.log(`${pass ? "OK" : "FAIL"}  ${q.id}  ${q.labelHe}`);
  console.log(`     שאלה: ${q.query}`);
  console.log(`     זרימה: ${q.expectPanelMode} | מנוע: ${engineQ}`);
  console.log(`     intents: ${intents.join(", ")}`);
  if (q.expectTitleKeywords?.length) {
    console.log(`     בדוק בכרטיסיות: ${q.expectTitleKeywords.join(" / ")}`);
  }
  console.log(`     ✓ ${q.verifyHe}`);
  console.log("");
}

if (failed) {
  console.error(`qa:news-chat — ${failed} routing failure(s)`);
  process.exit(1);
}

console.log(`qa:news-chat — all ${NEWS_ACCEPTANCE_QUERIES.length} routing checks OK`);
console.log("Next: npm run dev → paste each question in chat → confirm right panel + cards\n");
