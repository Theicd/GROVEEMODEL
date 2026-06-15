/** Direct probe of every search provider — writes PROVIDER_PROBE_REPORT.md */
import { writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { runWebSearch } from "../app/src/webSearch/orchestrator.ts";
import { ACCEPTANCE_QUERIES } from "../app/src/webSearch/acceptanceQueries.ts";

const __dir = dirname(fileURLToPath(import.meta.url));
const REPORT = join(__dir, "..", "PROVIDER_PROBE_REPORT.md");

type Probe = {
  provider: string;
  query: string;
  ok: boolean;
  error?: string;
  snippet: string;
  latencyMs?: number;
};

/** One acceptance query per provider id (best effort). */
const PROVIDER_SAMPLES: Record<string, string> = {};
for (const q of ACCEPTANCE_QUERIES) {
  for (const p of q.expectProvidersOk) {
    if (!PROVIDER_SAMPLES[p]) PROVIDER_SAMPLES[p] = q.query;
  }
}
PROVIDER_SAMPLES["searxng"] = "gaming industry news 2026";
PROVIDER_SAMPLES["stooq-commodity"] = "מה מחיר הנפט Brent?";
PROVIDER_SAMPLES["yahoo-finance"] = "מה מצב מדד S&P 500?";
PROVIDER_SAMPLES["wikipedia-en"] = "חפש מידע על Albert Einstein";
PROVIDER_SAMPLES["wikipedia-he"] = "חפש מידע על פירמידות";
PROVIDER_SAMPLES["huggingface-datasets"] = "hebrew dataset";
PROVIDER_SAMPLES["huggingface-models"] = "gemma models huggingface";
PROVIDER_SAMPLES["nager-holidays"] = "האם היום חג בגרמניה";
PROVIDER_SAMPLES["gdacs-disasters"] = "האם יש סופה פעילה באירופה?";
PROVIDER_SAMPLES["starlink-catalog"] = "כמה לווייני Starlink יש?";
PROVIDER_SAMPLES["celestrak"] = "כמה לוויינים פעילים בעולם?";
PROVIDER_SAMPLES["spacex-launches"] = "מתי השיגור הבא של SpaceX?";
PROVIDER_SAMPLES["noaa-space"] = "מה מזג האוויר החללי kp index?";
PROVIDER_SAMPLES["israel-alerts"] = "האם יש התראות בישראל?";
PROVIDER_SAMPLES["open-meteo-marine"] = "wave height Tel Aviv";
PROVIDER_SAMPLES["open-meteo-air-quality"] = "מה איכות האוויר בתל אביב?";

const ALL_PROVIDERS = [
  ...new Set([
    ...Object.keys(PROVIDER_SAMPLES),
    "open-meteo",
    "news-rss",
    "arxiv",
    "world-time",
    "usgs-earthquake",
    "adsb-aviation",
    "ais-ships",
    "osm-overpass-marine",
    "iss-tracker",
    "github",
    "nominatim-places",
    "osrm-distance",
    "rest-countries",
    "frankfurter-fx",
    "coingecko",
    "yahoo-finance",
    "wikidata-gov",
    "hacker-news",
    "url-context",
    "searxng",
    "wikipedia-en",
    "wikipedia-he",
  ]),
].sort();

const rows: Probe[] = [];

for (let i = 0; i < ALL_PROVIDERS.length; i++) {
  const provider = ALL_PROVIDERS[i];
  const query = PROVIDER_SAMPLES[provider] ?? "test";
  process.stdout.write(`\r[${i + 1}/${ALL_PROVIDERS.length}] ${provider}…`);
  try {
    const result = await runWebSearch(query);
    const src = result.sources.find((s) => s.provider === provider);
    if (src) {
      rows.push({
        provider,
        query,
        ok: src.ok && !!src.text.trim(),
        error: src.error,
        snippet: (src.text || src.error || "").split("\n")[0].slice(0, 90),
        latencyMs: src.latencyMs,
      });
    } else {
      rows.push({
        provider,
        query,
        ok: false,
        error: `not triggered (intents: ${result.intents.join(", ") || "—"})`,
        snippet: "",
      });
    }
  } catch (e) {
    rows.push({
      provider,
      query,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
      snippet: "",
    });
  }
  await new Promise((r) => setTimeout(r, 200));
}
console.log("\n");

const ok = rows.filter((r) => r.ok);
const fail = rows.filter((r) => !r.ok);

let md = `# דוח בדיקת מאגרי מידע — ${new Date().toISOString().slice(0, 19)}\n\n`;
md += `## סיכום\n\n`;
md += `- ✅ מחזירים מידע: **${ok.length}/${rows.length}**\n`;
md += `- ❌ נכשל / לא הופעל: **${fail.length}**\n\n`;

md += `## ✅ עובדים (${ok.length})\n\n`;
md += `| מקור | שאלת בדיקה | דוגמה | ms |\n|------|------------|-------|----|\n`;
for (const r of ok) {
  md += `| ${r.provider} | ${r.query.slice(0, 40)} | ${r.snippet.replace(/\|/g, "/")} | ${r.latencyMs ?? "—"} |\n`;
}

md += `\n## ❌ לא עובד / דורש הגדרה (${fail.length})\n\n`;
md += `| מקור | שאלת בדיקה | סיבה |\n|------|------------|------|\n`;
for (const r of fail) {
  md += `| ${r.provider} | ${r.query.slice(0, 40)} | ${(r.error ?? r.snippet).replace(/\|/g, "/").slice(0, 80)} |\n`;
}

writeFileSync(REPORT, md, "utf8");
console.log(`Report → ${REPORT}`);
console.log(`✅ ${ok.length}  ❌ ${fail.length}`);
