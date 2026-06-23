import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { StartupContext } from "../startupContext/types";

const WMO_ICON: Record<number, string> = {
  0: "☀️",
  1: "🌤",
  2: "⛅",
  3: "☁️",
  45: "🌫",
  48: "🌫",
  51: "🌦",
  61: "🌧",
  63: "🌧",
  65: "🌧",
  71: "🌨",
  80: "🌦",
  95: "⛈",
};

export function formatCableOsdDate(now: Date, uiLang: ChatUiLanguage, timezone?: string): string {
  const tz = timezone ?? "Asia/Jerusalem";
  const locale = uiLang === "he" ? "he-IL" : "en-US";
  try {
    return new Intl.DateTimeFormat(locale, {
      timeZone: tz,
      weekday: "short",
      day: "numeric",
      month: "long",
      year: "numeric",
    }).format(now);
  } catch {
    return now.toLocaleDateString(locale, { weekday: "short", day: "numeric", month: "long", year: "numeric" });
  }
}

export function formatCableOsdWeather(ctx: StartupContext | null): string | null {
  if (!ctx || ctx.localTempC == null) return null;
  const icon = WMO_ICON[ctx.localWeatherCode ?? 0] ?? "🌡";
  return `${icon} ${Math.round(ctx.localTempC)}°`;
}

/** Shorter label for the bottom OSD row. */
export function shortenCableChannelTitle(title: string, maxLen = 36): string {
  const trimmed = title
    .replace(/\s*\(\d+p\)/gi, "")
    .replace(/\s*\[[^\]]+\]/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
  if (trimmed.length <= maxLen) return trimmed;
  return `${trimmed.slice(0, maxLen - 1)}…`;
}
