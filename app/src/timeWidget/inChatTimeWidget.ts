import { isSinglePlaceTimeWidgetQuery } from "./resolveTimeWidget";
import type { TimeWidgetData } from "./types";

/** Chat should show only the clock card — no bullet text or search panel. */
export function isTimeWidgetOnlyTurn(text: string, widget: TimeWidgetData | null | undefined): boolean {
  return !!widget && isSinglePlaceTimeWidgetQuery(text);
}

export function buildTimeWidgetActivityDetail(widget: TimeWidgetData): string {
  try {
    const d = new Date(widget.anchorIso);
    const time = new Intl.DateTimeFormat("he-IL", {
      timeZone: widget.timezone,
      hour: "2-digit",
      minute: "2-digit",
    }).format(d);
    return `${widget.placeLabel} · ${time}`;
  } catch {
    return widget.placeLabel;
  }
}
