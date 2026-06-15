import { sanitizeSearchQuery } from "../webSearch/queryExtract";
import { extractTimeZonePair } from "../webSearch/queryExtract";
import { isWorldTimeQuery } from "../webSearch/intents";
import type { SearchSourceResult } from "../webSearch/types";
import { isLocalContextTimeQuery } from "../startupContext/localTime";
import type { StartupContext } from "../startupContext/types";
import type { TimeWidgetData } from "./types";

export const isSinglePlaceTimeWidgetQuery = (text: string): boolean => {
  const q = sanitizeSearchQuery(text);
  if (extractTimeZonePair(q) ?? extractTimeZonePair(text)) return false;
  return isLocalContextTimeQuery(q) || isWorldTimeQuery(q);
};

export const buildTimeWidgetFromStartupContext = (ctx: StartupContext): TimeWidgetData => {
  const place = ctx.cityName ? `${ctx.cityName}, ${ctx.countryName}` : ctx.countryName;
  return {
    placeLabel: place,
    timezone: ctx.timezone,
    anchorIso: ctx.datetime || new Date().toISOString(),
    utcOffsetLabel: ctx.utcOffset,
    dstActive: ctx.dst,
  };
};

export const buildTimeWidgetFromWorldTimeSource = (
  source: SearchSourceResult,
): TimeWidgetData | null => {
  if (source.provider !== "world-time" || !source.ok || !source.timeWidget) return null;
  return source.timeWidget;
};

export const buildShortTimeReply = (widget: TimeWidgetData): string => {
  try {
    const d = new Date(widget.anchorIso);
    const time = new Intl.DateTimeFormat("he-IL", {
      timeZone: widget.timezone,
      hour: "2-digit",
      minute: "2-digit",
    }).format(d);
    return `השעה ב-${widget.placeLabel}: ${time}.`;
  } catch {
    return `השעה ב-${widget.placeLabel}.`;
  }
};
