import { proxyAwareFetch } from "../proxyFetch";

export type HfTokenVerifyResult =
  | { ok: true; username: string }
  | { ok: false; message: string };

/** Check HF token via Hub whoami (does not use Inference Providers). */
export async function verifyHfToken(token: string): Promise<HfTokenVerifyResult> {
  const t = token.trim();
  if (!t) return { ok: false, message: "הזן HF Token" };
  if (!t.startsWith("hf_")) {
    return { ok: false, message: "פורמט לא תקין — Token צריך להתחיל ב-hf_" };
  }
  try {
    const response = await proxyAwareFetch("https://huggingface.co/api/whoami-v2", {
      method: "GET",
      headers: { Authorization: `Bearer ${t}` },
    });
    if (response.status === 401 || response.status === 403) {
      return { ok: false, message: "Token נדחה — בדוק שהעתקת נכון או צור token חדש" };
    }
    if (!response.ok) {
      return { ok: false, message: `Hub החזיר שגיאה ${response.status}` };
    }
    const json = (await response.json()) as { name?: string; fullname?: string };
    const username = (json.name || json.fullname || "").trim();
    if (!username) return { ok: false, message: "תשובה לא צפויה מ-Hub" };
    return { ok: true, username };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : "בדיקת חיבור נכשלה" };
  }
}
