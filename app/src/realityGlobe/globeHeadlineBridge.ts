import type { IntelHeadline, IntelTickerItem } from "./intelFeed";

export type GlobeHeadlineContext = {
  tickers: IntelTickerItem[];
  headlines: IntelHeadline[];
  countryCode?: string;
  countryName?: string;
};

type HeadlineRequest = {
  context: GlobeHeadlineContext;
  resolve: (headlines: IntelHeadline[]) => void;
};

const REQUEST_EVENT = "grovee-globe-headlines-request";
const RESULT_EVENT = "grovee-globe-headlines-result";

let pending: HeadlineRequest | null = null;

export function buildGlobeHeadlinePrompt(ctx: GlobeHeadlineContext): string {
  const top = ctx.tickers
    .slice(0, 12)
    .map((t) => `[${t.tag}] ${t.text}`)
    .join("\n");
  return `נתוני מוניטור עולמי חיים (${ctx.countryName || ctx.countryCode || "עולם"}):
${top || "אין אירועים חריגים כרגע"}

צור 3 כותרות חדשות קצרות בעברית — סגנון ערוץ חדשות מקצועי.
כל כותרת בשורה נפרדת, ללא מספור, ללא bullet.
התמקד בחריגות: רעידות, סופות, מטוסים צבאיים, לוויינים, צבע אדום, גלים, חלל.
אם אין חריגות — כותרת מצב שגרה קצרה.`;
}

export const GLOBE_HEADLINE_SYSTEM = `אתה עורך חדשות במוניטור REALITY LIVE.
כתוב כותרות קצרות, ברורות, בעברית — כמו BREAKING NEWS בטלוויזיה.
אסור הסברים, אסור JSON, אסור markdown — רק 3 שורות כותרות.`;

export function requestAiGlobeHeadlines(ctx: GlobeHeadlineContext): Promise<IntelHeadline[]> {
  return new Promise((resolve) => {
    pending = { context: ctx, resolve };
    window.dispatchEvent(new CustomEvent(REQUEST_EVENT, { detail: ctx }));
    window.setTimeout(() => {
      if (pending?.context === ctx) {
        pending = null;
        resolve([]);
      }
    }, 45_000);
  });
}

export function subscribeGlobeHeadlineRequests(
  onRequest: (ctx: GlobeHeadlineContext) => void,
): () => void {
  const handler = (e: Event) => {
    const detail = (e as CustomEvent<GlobeHeadlineContext>).detail;
    if (detail) onRequest(detail);
  };
  window.addEventListener(REQUEST_EVENT, handler);
  return () => window.removeEventListener(REQUEST_EVENT, handler);
}

export function publishGlobeHeadlineResult(headlines: IntelHeadline[]): void {
  pending?.resolve(headlines);
  pending = null;
  window.dispatchEvent(new CustomEvent(RESULT_EVENT, { detail: { headlines } }));
}

export function parseHeadlineLines(text: string): IntelHeadline[] {
  return text
    .split("\n")
    .map((l) => l.replace(/^[\d\.\-\*•]+\s*/, "").trim())
    .filter((l) => l.length > 8)
    .slice(0, 5)
    .map((line, i) => ({
      id: `ai-hl-${Date.now()}-${i}`,
      text: line,
      severity: line.match(/צבע אדום|דחוף|BREAKING|צונמי|M6|M7|מטוס.*צבא/i) ? 5 : 3,
    }));
}
