import type { SearchSourceResult } from "./types";
import { isWeatherQuery } from "./intents";

export function findOpenMeteoSource(sources: SearchSourceResult[]): SearchSourceResult | null {
  return sources.find((s) => s.provider === "open-meteo" && s.ok && s.text.trim()) ?? null;
}

/** Direct Hebrew reply from Open-Meteo — no LLM needed (avoids WebGPU OOM on simple weather turns). */
export function buildWeatherCannedReply(source: SearchSourceResult): string {
  const t = source.text;
  const line = (prefix: string): string => {
    const row = t.split("\n").find((l) => l.startsWith(prefix));
    return row?.slice(prefix.length).trim() ?? "";
  };

  const place = line("מיקום:");
  const desc = line("מצב:");
  const humidity = line("לחות:");
  const wind = line("רוח:");
  const tempRaw = line("טמפרטורה:") || line("טמפרatura:");
  const tempMatch = tempRaw.match(/([-\d.]+)°C\s*\(מרגיש\s+([-\d.]+)°C\)/);

  const temp = tempMatch?.[1] ?? tempRaw.replace(/°C.*/, "").trim();
  const feel = tempMatch?.[2] ?? "";

  let reply = place
    ? `🌡 **${place}** — עכשיו **${temp}°C**`
    : `🌡 עכשיו **${temp}°C**`;
  if (feel && feel !== "—") reply += ` (מרגיש כמו ${feel}°C)`;
  if (desc && desc !== "—") reply += `\n${desc}.`;
  if (humidity && humidity !== "—") reply += `\nלחות: ${humidity}.`;
  if (wind && wind !== "—") reply += `\nרוח: ${wind}.`;

  const forecastIdx = t.indexOf("תחזית 3 ימים:");
  if (forecastIdx >= 0) {
    reply += `\n\n${t.slice(forecastIdx).trim()}`;
  }

  reply +=
    "\n\n_(נתונים חיים מ-Open-Meteo — אותו סוג מקור כמו שכבת מזג האוויר במפה REALITY LIVE)_";
  return reply;
}

export function isPureWeatherTurn(query: string): boolean {
  const q = query.trim();
  if (!isWeatherQuery(q)) return false;
  if (/(?:רעיד|earthquake|מטוס|aircraft|github|reddit|arxiv|מחיר|stock)/i.test(q)) return false;
  return true;
}
