/**
 * Print NEWS topic QA checklist + validate routing for 25 «בנושא» queries.
 * Run: node --import tsx scripts/probe-topic-routing.ts
 */
import { classifySearchIntents } from "../app/src/webSearch/intents.ts";
import { needsWebSearch } from "../app/src/webSearch/intents.ts";
import { isTopicsOverviewQuery } from "../app/src/groveeNews/headlineIntent.ts";
import { normalizeNewsEngineQuery } from "../app/src/groveeNews/newsQueryNormalize.ts";
import { NEWS_TOPIC_ACCEPTANCE_QUERIES } from "../app/src/groveeNews/newsTopicAcceptanceQueries.ts";

let failed = 0;

console.log("\n=== GROVEE NEWS — topic query QA (25) ===\n");

for (const q of NEWS_TOPIC_ACCEPTANCE_QUERIES) {
  const intents = classifySearchIntents(q.query);
  const topics = isTopicsOverviewQuery(q.query);
  const panel = topics ? "topics" : "search";
  const engineQ = normalizeNewsEngineQuery(q.query);
  const intentOk = q.expectIntents.every((i) => intents.includes(i));
  const panelOk = panel === q.expectPanelMode;
  const engineOk = q.expectEngineQuery
    .toLowerCase()
    .split(/\s+/)
    .every((t) => engineQ.toLowerCase().includes(t));
  const webOk = needsWebSearch(q.query);
  const pass = intentOk && panelOk && engineOk && webOk;

  if (!pass) failed += 1;

  console.log(`${pass ? "OK" : "FAIL"}  ${q.id}`);
  console.log(`     שאלה: ${q.query}`);
  console.log(`     זרימה: ${panel} | מנוע: ${engineQ}`);
  console.log(`     בדוק בכרטיסיות: ${q.expectTitleKeywords.join(" / ")}`);
  console.log("");
}

if (failed) {
  console.error(`topic routing — ${failed} failure(s)`);
  process.exit(1);
}

console.log(`topic routing — all ${NEWS_TOPIC_ACCEPTANCE_QUERIES.length} OK`);
console.log("Next: npm run dev → paste each query → confirm Search panel + relevant cards\n");
