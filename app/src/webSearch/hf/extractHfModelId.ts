const HF_MODEL_URL = /huggingface\.co\/(?!spaces\/|datasets\/)([\w.-]+\/[\w.-]+)/i;
const HF_MODEL_REF = /\b([A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*)\b/;

/** Extract org/model from query or HF URL. */
export function extractHfModelIdFromQuery(query: string): string | null {
  const q = query.trim();
  if (!q) return null;
  const urlMatch = q.match(HF_MODEL_URL);
  if (urlMatch?.[1]) return urlMatch[1];
  const explicit = q.match(HF_MODEL_REF);
  if (explicit?.[1] && explicit[1].includes("/")) {
    const id = explicit[1];
    if (!/^(?:https?|www\.)/i.test(id)) return id;
  }
  return null;
}
