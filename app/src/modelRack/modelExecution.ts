import { buildPollinationsUrl } from "../cloudImage";
import { generateSdTurboPng } from "../localImageGen";
import { proxyAwareFetch } from "../webSearch/proxyFetch";
import { getHfToken } from "../webSearch/hf/hfModelSettings";
import { HF_INFERENCE_CHAT_URL } from "../webSearch/hf/hfModelTypes";
import type { RackModelEntry } from "./modelRack";
import {
  extractImageUrlFromGradioResult,
  extractTextFromGradioResult,
  runGradioPredict,
  spaceIdToHost,
} from "./gradioSpaceClient";

export type ExecuteResult =
  | { ok: true; content: string }
  | { ok: false; message: string };

function imageMarkdown(url: string, caption = "Generated"): string {
  return `![${caption}](${url})`;
}

async function executePollinations(model: RackModelEntry, prompt: string): Promise<ExecuteResult> {
  try {
    const url = buildPollinationsUrl({
      prompt,
      model: model.pollinationsModel ?? "flux",
    });
    return { ok: true, content: `${imageMarkdown(url)}\n\n*${model.label}*` };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : String(e) };
  }
}

async function executeSdTurboLocal(prompt: string, onStatus: (s: string) => void): Promise<ExecuteResult> {
  const out = await generateSdTurboPng(prompt, onStatus);
  if (!out.ok) return { ok: false, message: out.message };
  return {
    ok: true,
    content: `${imageMarkdown(out.objectUrl)}\n\n*SD-Turbo מקומי*`,
  };
}

async function executeHfImage(model: RackModelEntry, prompt: string): Promise<ExecuteResult> {
  const modelId = model.hfModelId;
  if (!modelId) return { ok: false, message: "חסר model id" };
  if (model.status === "token_required" && !getHfToken()) {
    return { ok: false, message: "מודל זה דורש HF Token — הוסף בהגדרות Gemma" };
  }

  const endpoint = `https://api-inference.huggingface.co/models/${encodeURIComponent(modelId)}`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getHfToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  try {
    const response = await proxyAwareFetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({ inputs: prompt }),
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      if (response.status === 401 || response.status === 403) {
        return { ok: false, message: "דורש HF Token — הגדרות Gemma" };
      }
      if (response.status === 503) {
        return { ok: false, message: "המודל בטעינה — נסה שוב בעוד דקה" };
      }
      return { ok: false, message: errText.slice(0, 200) || `HTTP ${response.status}` };
    }

    const blob = await response.blob();
    if (!blob.size) return { ok: false, message: "תשובה ריקה מהשרת" };
    const objectUrl = URL.createObjectURL(blob);
    return {
      ok: true,
      content: `${imageMarkdown(objectUrl)}\n\n*${modelId} · Hugging Face*`,
    };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : String(e) };
  }
}

async function executeHfChat(model: RackModelEntry, prompt: string): Promise<ExecuteResult> {
  const modelId = model.hfModelId;
  if (!modelId) return { ok: false, message: "חסר model id" };
  if (model.status === "token_required" && !getHfToken()) {
    return { ok: false, message: "מודל זה דורש HF Token — הוסף בהגדרות Gemma" };
  }

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getHfToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  try {
    const response = await proxyAwareFetch(HF_INFERENCE_CHAT_URL, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 1024,
      }),
    });
    const bodyText = await response.text();
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        return { ok: false, message: "דורש HF Token — הגדרות Gemma" };
      }
      return { ok: false, message: bodyText.slice(0, 240) || `HTTP ${response.status}` };
    }
    let parsed: { choices?: { message?: { content?: string } }[] };
    try {
      parsed = JSON.parse(bodyText) as typeof parsed;
    } catch {
      return { ok: false, message: "תשובה לא תקינה מה-API" };
    }
    const text = parsed.choices?.[0]?.message?.content?.trim();
    if (!text) return { ok: false, message: "תשובה ריקה מהמודל" };
    return { ok: true, content: text };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : String(e) };
  }
}

async function executeHfInference(model: RackModelEntry, prompt: string): Promise<ExecuteResult> {
  const modelId = model.hfModelId;
  if (!modelId) return { ok: false, message: "חסר model id" };

  const endpoint = `https://api-inference.huggingface.co/models/${encodeURIComponent(modelId)}`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getHfToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  try {
    const response = await proxyAwareFetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({ inputs: prompt }),
    });
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        return { ok: false, message: "דורש HF Token — הגדרות Gemma" };
      }
      const errText = await response.text().catch(() => "");
      return { ok: false, message: errText.slice(0, 240) || `HTTP ${response.status}` };
    }
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("image/") || contentType.includes("audio/") || contentType.includes("video/")) {
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const kind = contentType.includes("video/") ? "video" : contentType.includes("audio/") ? "audio" : "image";
      if (kind === "video") {
        return { ok: true, content: `<video controls src="${objectUrl}" className="msg-video"></video>\n\n*${modelId}*` };
      }
      if (kind === "audio") {
        return { ok: true, content: `<audio controls src="${objectUrl}"></audio>\n\n*${modelId}*` };
      }
      return { ok: true, content: `${imageMarkdown(objectUrl)}\n\n*${modelId}*` };
    }
    const text = await response.text();
    try {
      const json = JSON.parse(text) as unknown;
      return { ok: true, content: `\`\`\`json\n${JSON.stringify(json, null, 2).slice(0, 4000)}\n\`\`\`\n\n*${modelId}*` };
    } catch {
      return { ok: true, content: text.slice(0, 4000) || `*${modelId}*` };
    }
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : String(e) };
  }
}

async function executeHfGradioSpace(model: RackModelEntry, prompt: string): Promise<ExecuteResult> {
  const spaceId = model.hfSpaceId;
  const endpoint = model.gradioEndpoint;
  if (!spaceId || !endpoint) return { ok: false, message: "חסר מידע Space" };

  const host = spaceIdToHost(spaceId);
  const baseData = Array.isArray(model.gradioProbeData) ? [...model.gradioProbeData] : [];
  const data =
    baseData.length > 0
      ? baseData.map((v, i) => (i === 0 && typeof v === "string" ? prompt : v))
      : [prompt];

  const token = getHfToken();
  try {
    const result = await runGradioPredict(host, endpoint, data, token);
    if (!result?.length) {
      return { ok: false, message: "Space לא החזיר תוצאה — ייתכן מכסת ZeroGPU יומית" };
    }
    const imageUrl = extractImageUrlFromGradioResult(result);
    if (imageUrl) {
      return {
        ok: true,
        content: `${imageMarkdown(imageUrl)}\n\n*${spaceId} · HF Space*`,
      };
    }
    const text = extractTextFromGradioResult(result);
    if (text) return { ok: true, content: text };
    return { ok: false, message: "תשובה לא מזוהה מה-Space" };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : String(e) };
  }
}

/** Run generation for a non-Gemma rack model from chat input. */
export async function executeRackModel(
  model: RackModelEntry,
  prompt: string,
  onStatus: (s: string) => void,
): Promise<ExecuteResult> {
  const trimmed = prompt.trim();
  if (!trimmed) return { ok: false, message: "הכנס הוראה ליצירה" };

  if (model.status === "unavailable") {
    return { ok: false, message: "מודל זה לא זמין לשימוש — נסה לסרוק מחדש או לבחור מודל אחר" };
  }

  onStatus(`מריץ ${model.label}…`);

  switch (model.adapter) {
    case "pollinations":
      return executePollinations(model, trimmed);
    case "sd-turbo-local":
      return executeSdTurboLocal(trimmed, onStatus);
    case "hf-inference-image":
      return executeHfImage(model, trimmed);
    case "hf-inference":
      return executeHfInference(model, trimmed);
    case "hf-chat":
      return executeHfChat(model, trimmed);
    case "hf-gradio-space":
      return executeHfGradioSpace(model, trimmed);
    default:
      return { ok: false, message: "מודל לא נתמך להרצה ישירה" };
  }
}

export function rackModelNeedsGemma(model: RackModelEntry): boolean {
  return model.adapter === "gemma-local";
}

export function rackModelRunsInChat(model: RackModelEntry): boolean {
  return model.adapter !== "gemma-local";
}
