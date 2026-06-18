import { fetchJson } from "../fetchJson";
import { enrichHfModelsSearch } from "../hf/hfModelEnrich";
import { buildHuggingFaceSearchQuery } from "../intents";
import type { SearchSourceResult } from "../types";

export const fetchHuggingFaceModelsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "huggingface-models" as const;
  const label = "Hugging Face Models";
  const q = buildHuggingFaceSearchQuery(query) || query.trim().slice(0, 64);
  if (!q) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "שאילתה ריקה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
  try {
    const hfModelHits = await enrichHfModelsSearch(query);
    if (!hfModelHits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `אין מודלים עבור: ${q}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const text = hfModelHits
      .map((m) => {
        const tags = m.pipelineTag ? ` (${m.pipelineTag})` : "";
        const status = m.probed ? ` · ${m.status}` : "";
        return `- ${m.modelId}${tags} · ⬇${m.downloads ?? 0} · ♥${m.likes ?? 0}${status}\n  ${m.url}`;
      })
      .join("\n");

    return {
      provider,
      label,
      ok: true,
      text: `שאילתה: ${q}\n${text}`,
      url: `https://huggingface.co/models?search=${encodeURIComponent(q)}`,
      hfModelHits,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

export const fetchHuggingFaceDatasetsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "huggingface-datasets" as const;
  const label = "Hugging Face Datasets";
  const q = buildHuggingFaceSearchQuery(query) || query.trim().slice(0, 64);
  try {
    const data = await fetchJson<Array<{ id: string; downloads?: number; likes?: number }>>(
      `https://huggingface.co/api/datasets?search=${encodeURIComponent(q)}&limit=4&sort=downloads&direction=-1`,
    );
    if (!Array.isArray(data) || !data.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `אין datasets עבור: ${q}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const text = data
      .map((d) => `- ${d.id} · ⬇${d.downloads ?? 0}\n  https://huggingface.co/datasets/${d.id}`)
      .join("\n");
    return {
      provider,
      label,
      ok: true,
      text: `שאילתה: ${q}\n${text}`,
      url: `https://huggingface.co/datasets?search=${encodeURIComponent(q)}`,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
