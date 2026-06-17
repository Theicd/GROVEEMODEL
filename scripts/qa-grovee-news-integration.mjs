/**

 * Lightweight integration check for GROVEE-NEWS bridge (no browser).

 * Requires: npm run sync:news first.

 */

import { classifySearchIntents } from "../app/src/webSearch/intents.ts";

import { isTopicsOverviewQuery } from "../app/src/groveeNews/headlineIntent.ts";

import { normalizeNewsEngineQuery } from "../app/src/groveeNews/newsQueryNormalize.ts";



const checks = [

  ["headlineIntent world", isTopicsOverviewQuery("מה חדש בעולם?") === true],

  ["headlineIntent bare world", isTopicsOverviewQuery("מה קורה בעולם?") === true],

  ["headlineIntent specific", isTopicsOverviewQuery("חדשות על איראן") === false],

  ["intent news space he", classifySearchIntents("חפש חדשות על חלל").includes("news")],

  ["normalize space he", normalizeNewsEngineQuery("חפש חדשות על חלל") === "space"],

  ["normalize iran he", normalizeNewsEngineQuery("חדשות על איראן") === "iran"],

];



let failed = 0;

for (const [name, ok] of checks) {

  if (!ok) {

    console.error(`FAIL ${name}`);

    failed += 1;

  } else {

    console.log(`OK ${name}`);

  }

}



if (failed) process.exit(1);

console.log("\nqa-grovee-news-integration: bridge smoke OK");

