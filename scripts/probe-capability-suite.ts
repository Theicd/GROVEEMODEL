/** Capability QA — runs search + intent layer for full probe suite, writes markdown report. */
import { writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { CAPABILITY_PROBE_QUERIES } from "../app/src/capabilityProbeQueries.ts";
import { LANDING_CAPABILITY_CHIPS } from "../app/src/chatLandingContent.ts";
import { runWebSearch } from "../app/src/webSearch/orchestrator.ts";
import { classifySearchIntents, needsWebSearch } from "../app/src/webSearch/intents.ts";
import { buildGlobeCommand } from "../app/src/realityGlobe/intents.ts";
import { isGameSearchRequest, parseGameUserRequest } from "../app/src/gameSearch/gameIntents.ts";

const __dir = dirname(fileURLToPath(import.meta.url));
const REPORT_PATH = join(__dir, "..", "CAPABILITY_QA_REPORT.md");

type Status = "pass" | "partial" | "fail" | "manual" | "skip";

type Row = {
  id: string;
  category: string;
  query: string;
  tier: string;
  status: Status;
  detail: string;
  intents: string;
  providers: string;
};

function evaluate(probe: (typeof CAPABILITY_PROBE_QUERIES)[0], search: Awaited<ReturnType<typeof runWebSearch>> | null): Row {
  const intents = classifySearchIntents(probe.query);
  const intentStr = intents.join(", ") || "—";
  const okSources = search?.sources.filter((s) => s.ok) ?? [];
  const providerStr = okSources.map((s) => s.provider).join(", ") || "—";
  const snippet = okSources[0]?.text?.split("\n")[0]?.slice(0, 100) ?? "";

  if (probe.tier === "ui-globe") {
    const cmd = buildGlobeCommand(probe.query, intents);
    const ok = cmd !== null;
    return {
      id: probe.id,
      category: probe.category,
      query: probe.query,
      tier: probe.tier,
      status: ok ? "pass" : "fail",
      detail: ok
        ? `globe: ${cmd?.type}${cmd && "place" in cmd && cmd.place ? ` → ${cmd.place}` : ""}`
        : "לא זוהה פקודת מפה",
      intents: intentStr,
      providers: "—",
    };
  }

  if (probe.tier === "ui-game") {
    const gameOk = isGameSearchRequest(probe.query);
    const parsed = parseGameUserRequest(probe.query);
    return {
      id: probe.id,
      category: probe.category,
      query: probe.query,
      tier: probe.tier,
      status: gameOk ? "pass" : "fail",
      detail: gameOk ? `game query="${parsed.query}" cat=${parsed.category ?? "—"}` : "לא זוהה intent משחק",
      intents: intentStr,
      providers: "—",
    };
  }

  if (probe.tier === "unsupported") {
    const searched = needsWebSearch(probe.query);
    const hasData = okSources.length > 0;
    return {
      id: probe.id,
      category: probe.category,
      query: probe.query,
      tier: probe.tier,
      status: hasData ? "partial" : searched ? "fail" : "skip",
      detail: probe.notesHe ?? (hasData ? "unexpected data" : "צפוי — אין מקור"),
      intents: intentStr,
      providers: providerStr,
    };
  }

  if (probe.tier === "llm-synthesis") {
    const searched = needsWebSearch(probe.query);
    return {
      id: probe.id,
      category: probe.category,
      query: probe.query,
      tier: probe.tier,
      status: searched && okSources.length ? "partial" : "manual",
      detail: searched
        ? okSources.length
          ? `חיפוש חלקי (${okSources.length} מקורות) — דורש LLM לשילוב`
          : "needsWebSearch=true אך אין מקור — LLM בלבד"
        : "לא מפעיל חיפוש — תלוי LLM",
      intents: intentStr,
      providers: providerStr,
    };
  }

  if (!search) {
    return { id: probe.id, category: probe.category, query: probe.query, tier: probe.tier, status: "fail", detail: "no search", intents: intentStr, providers: "—" };
  }

  const intentMatch =
    !probe.expectIntents?.length || probe.expectIntents.some((i) => intents.includes(i));
  const hasOk = okSources.length > 0;

  if (probe.tier === "search-live") {
    const status: Status = hasOk && intentMatch ? "pass" : hasOk ? "partial" : "fail";
    return {
      id: probe.id,
      category: probe.category,
      query: probe.query,
      tier: probe.tier,
      status,
      detail: hasOk ? snippet : search.sources.find((s) => !s.ok)?.error ?? "אין מקורות",
      intents: intentStr,
      providers: providerStr,
    };
  }

  if (probe.tier === "search-partial" || probe.tier === "search-weak") {
    const status: Status = hasOk ? (intentMatch ? "pass" : "partial") : "fail";
    return {
      id: probe.id,
      category: probe.category,
      query: probe.query,
      tier: probe.tier,
      status,
      detail: hasOk ? snippet : probe.notesHe ?? "חלש/חלקי",
      intents: intentStr,
      providers: providerStr,
    };
  }

  return { id: probe.id, category: probe.category, query: probe.query, tier: probe.tier, status: "fail", detail: "unknown tier", intents: intentStr, providers: providerStr };
}

const rows: Row[] = [];
let i = 0;
for (const probe of CAPABILITY_PROBE_QUERIES) {
  i++;
  process.stdout.write(`\r[${i}/${CAPABILITY_PROBE_QUERIES.length}] ${probe.id}…`);
  let search = null;
  if (!["ui-globe", "ui-game", "unsupported"].includes(probe.tier) || probe.tier === "unsupported") {
    if (needsWebSearch(probe.query) || ["search-live", "search-partial", "search-weak", "llm-synthesis"].includes(probe.tier)) {
      try {
        search = await runWebSearch(probe.query);
      } catch (e) {
        search = { sources: [{ provider: "error", ok: false, error: String(e), text: "" }], intents: [], contextText: "", summaryHe: "" };
      }
    }
  }
  if (probe.tier === "ui-globe" || probe.tier === "ui-game") {
    rows.push(evaluate(probe, null));
  } else {
    rows.push(evaluate(probe, search));
  }
  await new Promise((r) => setTimeout(r, 120));
}
console.log("\n");

const counts = { pass: 0, partial: 0, fail: 0, manual: 0, skip: 0 };
for (const r of rows) counts[r.status]++;

const byCategory = new Map<string, Row[]>();
for (const r of rows) {
  const list = byCategory.get(r.category) ?? [];
  list.push(r);
  byCategory.set(r.category, list);
}

const icon = (s: Status) => ({ pass: "✅", partial: "⚠️", fail: "❌", manual: "🔵", skip: "⏭️" }[s]);

let md = `# דוח בדיקות יכולות GROVEE — ${new Date().toISOString().slice(0, 10)}\n\n`;
md += `## סיכום\n\n`;
md += `| סטטוס | כמות |\n|--------|------|\n`;
md += `| ✅ עובד (מקורות/Intent) | ${counts.pass} |\n`;
md += `| ⚠️ חלקי | ${counts.partial} |\n`;
md += `| ❌ נכשל | ${counts.fail} |\n`;
md += `| 🔵 ידני / LLM בלבד | ${counts.manual} |\n`;
md += `| ⏭️ לא נתמך (צפוי) | ${counts.skip} |\n`;
md += `\n**סה"כ שאלות:** ${rows.length}\n\n`;

md += `## מפת יכולות (לפי קטגוריה)\n\n`;
for (const [cat, list] of [...byCategory.entries()].sort((a, b) => a[0].localeCompare(b[0], "he"))) {
  const p = list.filter((x) => x.status === "pass").length;
  const t = list.length;
  md += `- **${cat}** — ${p}/${t} עובר\n`;
}

md += `\n## מה חוזר על עצמו (דפוסים)\n\n`;
md += `| דפוס | המלצה |\n|------|--------|\n`;
md += `| «הצג על הגלובוס/מפה» | ✅ Globe intent — עובד כשיש מקום/מדינה מפורש |\n`;
md += `| «כמה X באזור Y» (מטוסים/אוניות) | ✅ כש-Y במאגר bbox (ישראל, חיפה, רוטרדם) — ⚠️ לונדון/סואץ חלש |\n`;
md += `| «הכי עמוס / הכי גדול / Starlink» | ❌ אין API דירוג real-time |\n`;
md += `| שילוב 2+ מקורות (סופה+מטוסים) | 🔵 דורש LLM — חיפוש לא משלב לבד |\n`;
md += `| «שחק X» | ✅ Game panel — חלק מהניסוחים («שחק Doom») צריכים חידוד |\n`;
md += `| סקירת עולם / 20 אירועים | 🔵 LLM + partial search — לא תשובה מובנית אחת |\n`;
md += `| GitHub / HF / מזג / רעידות / ISS | ✅ מקורות חיים — תשובה תלויה ב-Gemma |\n\n`;

md += `## פירוט לפי שאלה\n\n`;
for (const [cat, list] of [...byCategory.entries()].sort((a, b) => a[0].localeCompare(b[0], "he"))) {
  md += `### ${cat}\n\n`;
  md += `| ID | שאלה | סטטוס | Intents | מקורות | הערה |\n`;
  md += `|----|------|--------|---------|---------|------|\n`;
  for (const r of list) {
    const q = r.query.replace(/\|/g, "\\|").slice(0, 55);
    const d = r.detail.replace(/\|/g, "\\|").slice(0, 60);
    md += `| ${r.id} | ${q} | ${icon(r.status)} | ${r.intents} | ${r.providers} | ${d} |\n`;
  }
  md += `\n`;
}

md += `## הצעות בממשק\n\n`;
md += `- **${CAPABILITY_PROBE_QUERIES.length}** שאלות בדיקה במערכת\n`;
md += `- **${LANDING_CAPABILITY_CHIPS.length}** הצעות ב-\`LANDING_CAPABILITY_CHIPS\` — **3** מתחלפות כל **10 שניות**\n\n`;

writeFileSync(REPORT_PATH, md, "utf8");
console.log(`Report → ${REPORT_PATH}`);
console.log(`✅ ${counts.pass}  ⚠️ ${counts.partial}  ❌ ${counts.fail}  🔵 ${counts.manual}  ⏭️ ${counts.skip}`);
process.exit(counts.fail > 15 ? 1 : 0);
