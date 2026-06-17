import { normalizeNewsEngineQuery, extractNewsTopicPhrase, isSpecificNewsTopicQuery } from "../app/src/groveeNews/newsQueryNormalize.ts";
import { isTopicsOverviewQuery } from "../app/src/groveeNews/headlineIntent.ts";
import { classifySearchIntents } from "../app/src/webSearch/intents.ts";
import { isTopHitRelevant } from "../app/src/groveeNews/engine/search/relevance.ts";
import { searchNews } from "../app/src/groveeNews/engine/engine/pipeline.ts";

const QUERIES = [
  { q: "חפש חדשות בנושא בינה מלאכותית", expect: ["artificial", "intelligence", "ai"] },
  { q: "חפש חדשות בנושא סייבר ואבטחת מידע", expect: ["cyber", "security"] },
  { q: "חפש חדשות בנושא פוליטיקה בישראל", expect: ["israel", "politic"] },
  { q: "חפש חדשות בנושא כלכלה ושוק ההון", expect: ["economy", "market", "stock"] },
  { q: "חפש חדשות בנושא טכנולוגיה וסטארטאפים", expect: ["technology", "startup"] },
  { q: "חפש חדשות בנושא מדע וחלל", expect: ["science", "space"] },
  { q: "חפש חדשות בנושא מזג אוויר קיצוני בעולם", expect: ["climate", "weather"] },
  { q: "חפש חדשות בנושא מלחמות וסכסוכים בעולם", expect: ["war", "conflict"] },
  { q: "חפש חדשות בנושא אנרגיה וחשמל ירוק", expect: ["energy", "renewable", "green"] },
  { q: "חפש חדשות בנושא תחבורה ורכב חשמלי", expect: ["car", "ev", "transport"] },
  { q: "חפש חדשות בנושא קריפטו וביטקוין", expect: ["crypto", "bitcoin"] },
  { q: "חפש חדשות בנושא חברות טכנולוגיה גדולות", expect: ["tech", "google", "apple", "microsoft"] },
  { q: "חפש חדשות בנושא רשתות חברתיות ואינטרנט", expect: ["social", "internet", "meta"] },
  { q: "חפש חדשות בנושא בריאות ורפואה", expect: ["health", "medical"] },
  { q: "חפש חדשות בנושא חינוך וטכנולוגיות למידה", expect: ["education", "learning"] },
  { q: "חפש חדשות בנושא חוק ורגולציה טכנולוגית", expect: ["regulation", "law", "tech"] },
  { q: "חפש חדשות בנושא צבא וביטחון", expect: ["military", "defense", "security"] },
  { q: "חפש חדשות בנושא חלל וחקר היקום", expect: ["space", "nasa"] },
  { q: "חפש חדשות בנושא גיימינג ותעשיית המשחקים", expect: ["gaming", "game"] },
  { q: "חפש חדשות בנושא תרבות ובידור", expect: ["culture", "entertainment", "film"] },
  { q: "חפש חדשות בנושא ספורט עולמי", expect: ["sport"] },
  { q: "חפש חדשות בנושא אירועים חריגים בעולם", expect: ["breaking", "disaster", "emergency"] },
  { q: "חפש חדשות בנושא חדשנות ובינה מלאכותית יישומית", expect: ["ai", "innovation"] },
  { q: "חפש חדשות בנושא סטארטאפים ישראליים", expect: ["startup", "israel"] },
  { q: "חפש חדשות בנושא חקלאות וטכנולוגיות מזון", expect: ["agriculture", "food"] },
];

let routingFails = 0;
let relevanceFails = 0;

console.log("\n=== Topic query probe (routing + relevance) ===\n");

for (const { q, expect } of QUERIES) {
  const phrase = extractNewsTopicPhrase(q);
  const engine = normalizeNewsEngineQuery(q);
  const specific = isSpecificNewsTopicQuery(q);
  const topics = isTopicsOverviewQuery(q);
  const intents = classifySearchIntents(q);

  const routingOk =
    intents.includes("news") && !topics && specific && engine.length > 0;

  let hits = [];
  let relevant = 0;
  if (engine) {
    hits = await searchNews(engine);
    if (specific) {
      hits = hits.filter((h) => isTopHitRelevant(h.article, engine));
    }
    relevant = hits.length;
  }

  const top3 = hits.slice(0, 3).map((h) => h.article.title?.slice(0, 70) ?? "?");
  const titleBlob = top3.join(" ").toLowerCase();
  const keywordHit = expect.some((kw) => engine.toLowerCase().includes(kw) || titleBlob.includes(kw));

  const pass = routingOk && relevant > 0 && keywordHit;
  if (!routingOk) routingFails += 1;
  if (relevant === 0 || !keywordHit) relevanceFails += 1;

  console.log(`${pass ? "OK" : "FAIL"}  ${q}`);
  console.log(`     phrase: ${phrase}`);
  console.log(`     engine: ${engine || "(empty)"} | panel: ${topics ? "topics" : "search"} | hits: ${relevant}`);
  if (top3.length) console.log(`     top: ${top3[0]}`);
  console.log("");
}

console.log(`Routing failures: ${routingFails}`);
console.log(`Relevance failures: ${relevanceFails}`);
