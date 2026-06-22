/**
 * Probe Grove Search Companion (OpenSERP) — health + mega search.
 * Usage: node scripts/probe-search-companion.mjs
 */
const BASE = process.env.OPENSERP_URL ?? "http://127.0.0.1:7000";

async function main() {
  console.log(`Probing ${BASE} ...`);

  const healthRes = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(5000) });
  const health = await healthRes.json();
  console.log("Health:", healthRes.status, health.status ?? health);

  const q = new URLSearchParams({
    text: "webgpu browser ai",
    engines: "google,bing,duck",
    limit: "3",
    mode: "any",
    format: "json",
    dedupe: "true",
    merge: "true",
  });
  const searchRes = await fetch(`${BASE}/mega/search?${q}`, { signal: AbortSignal.timeout(45000) });
  const search = await searchRes.json();
  const hits = search.results ?? [];
  console.log("Search:", searchRes.status, `${hits.length} hits`);
  for (const [i, h] of hits.slice(0, 3).entries()) {
    console.log(`  ${i + 1}. ${h.title?.slice(0, 60)} — ${h.url}`);
  }
  if (!hits.length) {
    console.error("FAIL: no search results");
    process.exit(1);
  }
  console.log("OK");
}

main().catch((err) => {
  console.error("FAIL:", err.message);
  process.exit(1);
});
