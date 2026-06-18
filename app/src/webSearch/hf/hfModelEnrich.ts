import { fetchJson } from "../fetchJson";
import { buildHuggingFaceSearchQuery, isHuggingFaceImageQuery } from "../intents";
import {
  fetchWorkingModelsFromScanner,
  filterScannerModelsByQuery,
  isHfScannerAvailable,
  scannerRowToProbe,
  testModelViaScanner,
  type ScannerModelRow,
} from "./hfApiScannerClient";
import { buildHfCurlSnippet, buildHfPythonSnippet } from "./hfConnectionSnippets";
import { extractHfModelIdFromQuery } from "./extractHfModelId";
import { getHfToken } from "./hfModelSettings";
import { probeHfModelInBrowser, STATUS_WORKING } from "./hfModelProbe";
import {
  hubModelId,
  type HfHubModelSummary,
  type HfModelSerpHit,
  type HfProbeResult,
} from "./hfModelTypes";

const MAX_PROBE = 2;

function classifyCategory(pipeline?: string, modelId = ""): string {
  const p = (pipeline || "").toLowerCase();
  if (p.includes("text-to-image") || p.includes("image-to-image")) return "Image";
  if (p.includes("video")) return "Video";
  if (p.includes("image") || p.includes("vision")) return "Vision";
  if (/coder|code|starcoder/i.test(modelId)) return "Code";
  return "Text";
}

function hubRowToHit(m: HfHubModelSummary, probe: HfProbeResult | null, probeSource: HfModelSerpHit["probeSource"]): HfModelSerpHit {
  const modelId = hubModelId(m);
  const url = `https://huggingface.co/${modelId}`;
  const pipeline = m.pipeline_tag;
  const endpoint = probe?.endpoint || "https://router.huggingface.co/v1/chat/completions";
  const status = probe?.status || "NOT PROBED";
  const snippetParts = [
    pipeline ? `Pipeline: ${pipeline}` : "",
    probe?.provider ? `Provider: ${probe.provider}` : "",
    probe?.accessMode ? `Access: ${probe.accessMode}` : "",
    probe?.latency != null ? `Latency: ${probe.latency}s` : "",
    `⬇ ${(m.downloads ?? 0).toLocaleString()} · ♥ ${m.likes ?? 0}`,
  ].filter(Boolean);

  return {
    id: `hf-${modelId.replace(/\//g, "--")}`,
    modelId,
    url,
    title: modelId,
    snippet: snippetParts.join(" · "),
    pipelineTag: pipeline,
    category: classifyCategory(pipeline, modelId),
    organization: modelId.split("/")[0],
    downloads: m.downloads,
    likes: m.likes,
    status,
    provider: probe?.provider || "Unknown",
    accessMode: probe?.accessMode || "UNKNOWN",
    latency: probe?.latency,
    endpoint,
    curlSnippet: buildHfCurlSnippet(modelId, endpoint),
    pythonSnippet: buildHfPythonSnippet(modelId, endpoint),
    probed: !!probe,
    probeSource,
    errorText: probe?.errorText,
  };
}

function scannerRowToHit(row: ScannerModelRow): HfModelSerpHit {
  const probe = scannerRowToProbe(row);
  const modelId = row.model_id;
  const endpoint = probe.endpoint;
  return {
    id: `hf-${modelId.replace(/\//g, "--")}`,
    modelId,
    url: `https://huggingface.co/${modelId}`,
    title: modelId,
    snippet: [
      row.pipeline ? `Pipeline: ${row.pipeline}` : "",
      row.category ? `Category: ${row.category}` : "",
      `Provider: ${probe.provider}`,
      `Access: ${probe.accessMode}`,
      probe.latency != null ? `Latency: ${probe.latency}s` : "",
      `⬇ ${(row.downloads ?? 0).toLocaleString()}`,
    ]
      .filter(Boolean)
      .join(" · "),
    pipelineTag: row.pipeline,
    category: row.category || classifyCategory(row.pipeline, modelId),
    organization: row.organization || modelId.split("/")[0],
    sizeParam: row.size_param,
    downloads: row.downloads,
    likes: row.likes,
    status: probe.status,
    provider: probe.provider,
    accessMode: probe.accessMode,
    latency: probe.latency,
    endpoint,
    curlSnippet: buildHfCurlSnippet(modelId, endpoint),
    pythonSnippet: buildHfPythonSnippet(modelId, endpoint),
    probed: true,
    probeSource: "scanner",
    errorText: probe.errorText,
  };
}

async function probeModel(modelId: string): Promise<{ probe: HfProbeResult; source: HfModelSerpHit["probeSource"] }> {
  const token = getHfToken();
  if (await isHfScannerAvailable()) {
    const scanned = await testModelViaScanner(modelId, token);
    if (scanned) return { probe: scanned, source: "scanner" };
  }
  const browser = await probeHfModelInBrowser(modelId, token);
  return { probe: browser, source: "browser" };
}

async function fetchHubModels(q: string, limit: number): Promise<HfHubModelSummary[]> {
  const imageModels = isHuggingFaceImageQuery(q);
  const pipeline = imageModels ? "&pipeline_tag=text-to-image" : "";
  const data = await fetchJson<HfHubModelSummary[]>(
    `https://huggingface.co/api/models?search=${encodeURIComponent(q)}&limit=${limit}&sort=downloads&direction=-1${pipeline}`,
    { headers: { "User-Agent": "GROVEEMODEL/1.0" } },
    { timeoutMs: 14_000 },
  );
  return Array.isArray(data) ? data : [];
}

function sortHits(hits: HfModelSerpHit[]): HfModelSerpHit[] {
  const weight = (s: string) => {
    const u = s.toUpperCase();
    if (u === STATUS_WORKING) return 5;
    if (u === "LOADING") return 4;
    if (u === "AVAILABLE BUT RESTRICTED") return 3;
    if (u === "RATE LIMITED") return 2;
    if (u === "PROVIDER REQUIRED") return 1;
    return 0;
  };
  return [...hits].sort(
    (a, b) =>
      weight(b.status) - weight(a.status) ||
      (b.downloads ?? 0) - (a.downloads ?? 0) ||
      (a.latency ?? 999) - (b.latency ?? 999),
  );
}

/** Hub search + API probe (scanner first, browser fallback) + working-model cache. */
export async function enrichHfModelsSearch(query: string): Promise<HfModelSerpHit[]> {
  const q = buildHuggingFaceSearchQuery(query) || query.trim().slice(0, 64);
  if (!q) return [];

  const explicitId = extractHfModelIdFromQuery(query);
  const hits: HfModelSerpHit[] = [];
  const seen = new Set<string>();

  const pushHit = (hit: HfModelSerpHit) => {
    if (seen.has(hit.modelId)) return;
    seen.add(hit.modelId);
    hits.push(hit);
  };

  if (explicitId) {
    const { probe, source } = await probeModel(explicitId);
    let hub: HfHubModelSummary = { id: explicitId };
    try {
      const meta = await fetchJson<HfHubModelSummary>(
        `https://huggingface.co/api/models/${encodeURIComponent(explicitId)}`,
        { headers: { "User-Agent": "GROVEEMODEL/1.0" } },
        { timeoutMs: 12_000 },
      );
      hub = meta;
    } catch {
      /* metadata optional */
    }
    pushHit(hubRowToHit(hub, probe, source));
  }

  const working = filterScannerModelsByQuery(await fetchWorkingModelsFromScanner(300), q);
  const workingLimit = (await isHfScannerAvailable()) ? 6 : 4;
  for (const row of working.slice(0, workingLimit)) pushHit(scannerRowToHit(row));

  const hubList = await fetchHubModels(q, 10);
  for (const m of hubList) {
    const id = hubModelId(m);
    if (!id || seen.has(id)) continue;
    pushHit(hubRowToHit(m, null, "none"));
    if (hits.length >= 10) break;
  }

  const probeTargets = hubList
    .map(hubModelId)
    .filter((id) => id && seen.has(id))
    .slice(0, MAX_PROBE);

  if (probeTargets.length) {
    const probed = await Promise.all(
      probeTargets.map(async (modelId) => {
        const { probe, source } = await probeModel(modelId);
        return { modelId, probe, source };
      }),
    );
    for (const row of probed) {
      const idx = hits.findIndex((h) => h.modelId === row.modelId);
      if (idx < 0) continue;
      const hub = hubList.find((h) => hubModelId(h) === row.modelId) || { id: row.modelId };
      hits[idx] = hubRowToHit(hub, row.probe, row.source);
    }
  }

  return sortHits(hits).slice(0, 10);
}
