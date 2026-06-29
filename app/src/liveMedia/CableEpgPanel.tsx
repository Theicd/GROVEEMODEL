import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "../searchResults/types";
import { isWebOnlyHost } from "../webSearch/proxyFetch";
import { loadEpgGuide, verifyEpgFeedsLoaded, type EpgGuideEntry } from "./epg/epgGuideStore";
import {
  buildEpgGridWindow,
  EPG_GRID_ROW_PX,
  EPG_GRID_SLOT_PX,
  formatGridRange,
  formatGridTime,
  layoutProgramsInWindow,
} from "./epg/epgGridLayout";
import { getStreamEpgOffset, shiftEpgPrograms } from "./epg/epgUtcOffset";
import {
  ALL_USER_CATEGORIES,
  USER_CHANNEL_CATEGORIES,
  categoryLabelEn,
  categoryLabelHe,
  type UserChannelCategory,
} from "./channelUserTaxonomy";
import type { ChannelDisplayMeta } from "./channelDisplay";

type Props = {
  favorites: UnifiedSearchHit[];
  categoryOrder?: UserChannelCategory[];
  focusHit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
  onClose: () => void;
};

function shortenTitle(title: string, max = 42): string {
  const t = title.trim();
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

export function CableEpgPanel({ favorites, categoryOrder, focusHit, uiLang, onClose }: Props) {
  const rtl = uiLang === "he";
  const [entries, setEntries] = useState<EpgGuideEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<"feed" | "nomatch" | false>(false);
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [now, setNow] = useState(() => new Date());
  const scrollRef = useRef<HTMLDivElement>(null);

  const webOnly = isWebOnlyHost();
  const L =
    uiLang === "he"
      ? {
          title: "לוח שידורים",
          loading: (n: number, t: number) => `טוען לוח שידורים… ${n}/${t}`,
          feedError: webOnly
            ? "לא הצלחנו לטעון את נתוני ה-EPG מהרשת. בדוק חיבור לאינטרנט ולחץ נסה שוב."
            : "לא הצלחנו לטעון את נתוני ה-EPG מהרשת. עצור והפעל מחדש את npm run dev, ואז רענן.",
          noMatch: "נתוני EPG נטענו, אך לא נמצאו תוכניות לערוצים המועדפים בטווח הזמן הזה.",
          retry: "נסה שוב",
          channels: (n: number, total: number) => `${n} ערוצים עם תוכניות · ${total} סה״כ`,
          noShows: "אין תוכניות בטווח הזמן",
          close: "סגור",
          today: "היום",
          scrollUp: "למעלה",
          scrollDown: "למטה",
        }
      : {
          title: "TV Guide",
          loading: (n: number, t: number) => `Loading TV guide… ${n}/${t}`,
          feedError: webOnly
            ? "Could not load EPG data from the network. Check your connection and tap Retry."
            : "Could not load EPG data. Stop and restart npm run dev, then refresh.",
          noMatch: "EPG feeds loaded, but no programmes matched your favorites in this time window.",
          retry: "Retry",
          channels: (n: number, total: number) => `${n} channels with shows · ${total} total`,
          noShows: "No shows in this time range",
          close: "Close",
          today: "Today",
          scrollUp: "Up",
          scrollDown: "Down",
        };

  const scrollBy = useCallback((delta: number) => {
    scrollRef.current?.scrollBy({ top: delta, behavior: "smooth" });
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const feedsOk = await verifyEpgFeedsLoaded();
      if (!feedsOk) {
        setLoadError("feed");
        return;
      }
      const result = await loadEpgGuide(favorites, (partial, loaded, total) => {
        setEntries(partial);
        setProgress({ loaded, total });
      });
      setEntries(result);
      const withData = result.filter((e) => (e.schedule?.programs.length ?? 0) > 0).length;
      if (!withData) setLoadError("nomatch");
    } catch {
      setLoadError("feed");
    } finally {
      setLoading(false);
    }
  }, [favorites]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "PageDown" && e.key !== "PageUp") {
        return;
      }
      const el = scrollRef.current;
      if (!el) return;
      e.preventDefault();
      e.stopPropagation();
      const delta = e.key === "ArrowUp" || e.key === "PageUp" ? -160 : 160;
      el.scrollBy({ top: delta, behavior: "smooth" });
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  useEffect(() => {
    if (loading) return;
    const row = scrollRef.current?.querySelector(".lm-cable-epg-grid-row.is-focus");
    row?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [focusHit.id, loading]);

  const displayCategoryOrder = categoryOrder?.length ? categoryOrder : [...ALL_USER_CATEGORIES];
  const gridWindow = useMemo(() => buildEpgGridWindow(now), [now]);
  const withData = entries.filter((e) => (e.schedule?.programs.length ?? 0) > 0);
  const rows = withData.length > 0 ? withData : entries;
  const focusRowId = focusHit.id;

  const rowsByCategory = useMemo(() => {
    const grouped = new Map<UserChannelCategory, EpgGuideEntry[]>();
    for (const cat of USER_CHANNEL_CATEGORIES) grouped.set(cat.id, []);
    for (const entry of rows) {
      const cat = (entry.hit.meta as ChannelDisplayMeta | undefined)?.userCategory ?? "general";
      const list = grouped.get(cat) ?? [];
      list.push(entry);
      grouped.set(cat, list);
    }
    return grouped;
  }, [rows]);

  const renderEpgRow = (entry: EpgGuideEntry) => {
    const streamUrl = entry.hit.mediaPlayUrl || entry.hit.url;
    const offset = entry.hit.id === focusRowId ? getStreamEpgOffset(streamUrl) : 0;
    const programs = entry.schedule?.programs ? shiftEpgPrograms(entry.schedule.programs, offset) : [];
    const blocks = programs.length ? layoutProgramsInWindow(programs, gridWindow, now) : [];
    const isFocus = entry.hit.id === focusRowId;
    return (
      <div key={entry.hit.id} className={`lm-cable-epg-grid-row${isFocus ? " is-focus" : ""}`}>
        <div className="lm-cable-epg-grid-label" title={entry.hit.title}>
          {entry.hit.imageUrl ? (
            <img src={entry.hit.imageUrl} alt="" className="lm-cable-epg-grid-logo" />
          ) : (
            <span className="lm-cable-epg-grid-logo lm-cable-epg-grid-logo--text">
              {entry.hit.title.slice(0, 2).toUpperCase()}
            </span>
          )}
          <span className="lm-cable-epg-grid-label-text">{shortenTitle(entry.hit.title, 28)}</span>
        </div>
        <div className="lm-cable-epg-grid-track" style={{ width: gridWindow.totalPx, height: EPG_GRID_ROW_PX }}>
          {blocks.length === 0 ? (
            <div className="lm-cable-epg-grid-empty">{L.noShows}</div>
          ) : (
            blocks.map((block) => (
              <div
                key={`${block.program.start.toISOString()}-${block.program.title}`}
                className={`lm-cable-epg-grid-block${block.live ? " is-live" : ""}`}
                style={{ left: block.leftPx, width: block.widthPx }}
                title={`${block.program.title}\n${formatGridRange(block.program.start, block.program.end, rtl)}`}
              >
                <span className="lm-cable-epg-grid-block-title">{shortenTitle(block.program.title, 36)}</span>
                <span className="lm-cable-epg-grid-block-time">
                  {formatGridRange(block.program.start, block.program.end, rtl)}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="lm-cable-epg-backdrop lm-cable-epg-backdrop--grid" onClick={onClose} role="presentation">
      <div
        className="lm-cable-epg lm-cable-epg--grid"
        role="dialog"
        aria-modal="true"
        aria-labelledby="lm-cable-epg-title"
        dir={rtl ? "rtl" : "ltr"}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="lm-cable-epg-head">
          <div>
            <p className="lm-cable-epg-kicker">EPG</p>
            <h3 id="lm-cable-epg-title" className="lm-cable-epg-title">
              {L.title}
            </h3>
            {!loading && !loadError ? (
              <p className="lm-cable-epg-source">{L.channels(withData.length, entries.length)}</p>
            ) : null}
          </div>
          <div className="lm-cable-epg-head-actions">
            <button type="button" className="lm-cable-osd-btn" onClick={() => scrollBy(-160)} aria-label={L.scrollUp}>
              ▲
            </button>
            <button type="button" className="lm-cable-osd-btn" onClick={() => scrollBy(160)} aria-label={L.scrollDown}>
              ▼
            </button>
            <span className="lm-cable-epg-date-pill">{L.today}</span>
            <button type="button" className="lm-cable-osd-btn" onClick={onClose}>
              {L.close}
            </button>
          </div>
        </header>

        {loading ? (
          <p className="lm-cable-epg-status">{L.loading(progress.loaded, progress.total || favorites.length)}</p>
        ) : null}
        {loadError ? (
          <div className="lm-cable-epg-status lm-cable-epg-status--error">
            <p>{loadError === "feed" ? L.feedError : L.noMatch}</p>
            <button type="button" className="lm-cable-osd-btn" onClick={() => void load()}>
              {L.retry}
            </button>
          </div>
        ) : null}

        {!loading && !loadError && rows.length > 0 ? (
          <div className="lm-cable-epg-grid-scroll" ref={scrollRef}>
            <div
              className="lm-cable-epg-grid"
              style={{ "--epg-track-width": `${gridWindow.totalPx}px`, "--epg-row-height": `${EPG_GRID_ROW_PX}px` } as CSSProperties}
            >
              <div className="lm-cable-epg-grid-corner" />
              <div className="lm-cable-epg-grid-times" style={{ width: gridWindow.totalPx }}>
                {gridWindow.slots.map((slot) => (
                  <div
                    key={slot.toISOString()}
                    className="lm-cable-epg-grid-slot"
                    style={{ width: EPG_GRID_SLOT_PX }}
                  >
                    {formatGridTime(slot, rtl)}
                  </div>
                ))}
              </div>

              {displayCategoryOrder.map((catId) => {
                const sectionRows = rowsByCategory.get(catId) ?? [];
                if (!sectionRows.length) return null;
                return (
                  <div key={catId} className="lm-cable-epg-category-block">
                    <div className="lm-cable-epg-category-header">
                      {rtl ? categoryLabelHe(catId) : categoryLabelEn(catId)} · {sectionRows.length}
                    </div>
                    {sectionRows.map((entry) => renderEpgRow(entry))}
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
