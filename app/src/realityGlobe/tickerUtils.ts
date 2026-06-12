import type { IntelHeadline, IntelTickerItem } from "./intelFeed";

const MS_24H = 24 * 60 * 60 * 1000;

export function formatRelativeTimeHe(ts: number, now = Date.now()): string {
  const diff = Math.max(0, now - ts);
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "עכשיו";
  if (mins < 60) return `לפני ${mins} דק'`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `לפני ${hours} ש'`;
  const days = Math.floor(hours / 24);
  return `לפני ${days} ימים`;
}

export function formatClockHe(ts: number): string {
  return new Date(ts).toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
}

export function iconForCategory(tag: string, category?: string): string {
  const key = `${category || ""} ${tag}`.toUpperCase();
  if (/צבע|RED|ISRAEL|IL/.test(key)) return "🚨";
  if (/רעיד|SEISMIC|EQ|RICHTER|TSUNAMI/.test(key)) return "🌍";
  if (/סופ|WEATHER|WIND|רוח|חום|קור/.test(key)) return "🌪";
  if (/ים|MARINE|WAVE|גל/.test(key)) return "🌊";
  if (/חלל|SPACE|KP|SOLAR/.test(key)) return "☄";
  if (/AVIATION|מטוס|AIR/.test(key)) return "✈";
  if (/SHIP|ספינ|MARITIME/.test(key)) return "⛴";
  if (/SAT|לווי|ISS/.test(key)) return "🛰";
  if (/INSIGHT|AI|CORREL/.test(key)) return "🤖";
  if (/BREAKING|דחוף/.test(key)) return "🔴";
  if (/תנוע|TRAFFIC|LIVE/.test(key)) return "📡";
  return "◆";
}

export function mergeTickerHistory(
  incoming: IntelTickerItem[],
  store: Map<string, IntelTickerItem>,
  now = Date.now(),
): IntelTickerItem[] {
  const cutoff = now - MS_24H;
  for (const item of incoming) {
    if (!item?.id) continue;
    const ts = item.ts ?? now;
    store.set(item.id, { ...item, ts, icon: item.icon || iconForCategory(item.tag, item.category) });
  }
  for (const [id, item] of store) {
    if ((item.ts ?? 0) < cutoff) store.delete(id);
  }
  return [...store.values()].sort(
    (a, b) => b.severity - a.severity || (b.ts ?? 0) - (a.ts ?? 0),
  );
}

export function buildUnifiedTickerLine(
  items: IntelTickerItem[],
  headlines: IntelHeadline[],
  now = Date.now(),
): IntelTickerItem[] {
  const line: IntelTickerItem[] = [];

  for (const h of headlines.slice(0, 6)) {
    line.push({
      id: `hl-${h.id}`,
      severity: h.severity,
      tag: h.severity >= 5 ? "BREAKING" : "כותרת",
      text: h.text,
      time: formatClockHe(now),
      ts: now,
      icon: h.severity >= 5 ? "🔴" : "📰",
      category: "HEADLINE",
    });
  }

  for (const item of items) {
    line.push({
      ...item,
      ts: item.ts ?? now,
      icon: item.icon || iconForCategory(item.tag, item.category),
      time: item.time || formatClockHe(item.ts ?? now),
    });
  }

  if (!line.length) {
    line.push({
      id: "idle-live",
      severity: 1,
      tag: "LIVE",
      text: "REALITY LIVE · מנטר עולם · ממתין לנתונים…",
      time: formatClockHe(now),
      ts: now,
      icon: "📡",
      category: "LIVE",
    });
  }

  const seen = new Set<string>();
  return line.filter((item) => {
    const key = `${item.tag}:${item.text}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function tickerDurationSec(count: number): number {
  return Math.min(180, Math.max(70, count * 9));
}
