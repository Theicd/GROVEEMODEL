export async function probeTmdbConnection(apiKey: string): Promise<{ ok: boolean; message: string }> {
  const key = apiKey.trim();
  if (!key) return { ok: false, message: "לא הוזן מפתח TMDB" };
  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/configuration?api_key=${encodeURIComponent(key)}`,
    );
    if (!res.ok) {
      return { ok: false, message: `TMDB דחה את המפתח (HTTP ${res.status})` };
    }
    const search = await fetch(
      `https://api.themoviedb.org/3/search/movie?api_key=${encodeURIComponent(key)}&query=Inception`,
    );
    if (!search.ok) {
      return { ok: true, message: "מפתח תקין — חיבור ל-TMDB הצליח" };
    }
    const data = (await search.json()) as { results?: unknown[] };
    const n = data.results?.length ?? 0;
    return { ok: true, message: `מפתח תקין — TMDB מחזיר נתונים (דוגמה: ${n} תוצאות ל-Inception)` };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : "שגיאת רשת ל-TMDB" };
  }
}
