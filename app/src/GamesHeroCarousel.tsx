import { useCallback, useEffect, useState } from "react";
import type { OnlineGame } from "./gameSearch/types";
import { gameHeroImageUrl } from "./gameSearch/gameThumbnail";

const ROTATE_MS = 6500;

type Props = {
  games: OnlineGame[];
  favoriteCount?: number;
  layout?: "side" | "full";
  onPlay: (game: OnlineGame) => void;
};

const MAX_HERO_DOTS = 12;

export function GamesHeroCarousel({ games, favoriteCount, layout = "full", onPlay }: Props) {
  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);
  const total = games.length;
  const current = games[index] ?? null;

  const goTo = useCallback(
    (next: number) => {
      if (total < 1) return;
      const idx = ((next % total) + total) % total;
      setFade(false);
      window.setTimeout(() => {
        setIndex(idx);
        setFade(true);
      }, 220);
    },
    [total],
  );

  useEffect(() => {
    if (total < 2) return;
    const id = window.setInterval(() => {
      setFade(false);
      window.setTimeout(() => {
        setIndex((i) => (i + 1) % total);
        setFade(true);
      }, 220);
    }, ROTATE_MS);
    return () => window.clearInterval(id);
  }, [total]);

  useEffect(() => {
    setIndex(0);
    setFade(true);
  }, [games]);

  if (!current || total < 1) return null;

  const heroSrc = gameHeroImageUrl(current);
  const countLabel = favoriteCount ?? total;
  const showDots = total <= MAX_HERO_DOTS;

  return (
    <section
      className={`games-hero${layout === "side" ? " games-hero--compact" : ""}`}
      dir="rtl"
      aria-label="מועדפים — רוטציה בהדר"
    >
      <div className={`games-hero-stage${fade ? " is-visible" : ""}`}>
        <img
          key={`bg-${current.id}`}
          className="games-hero-bg"
          src={heroSrc}
          alt=""
          loading="eager"
          referrerPolicy="no-referrer"
        />
        <div className="games-hero-bg-blur" aria-hidden="true" style={{ backgroundImage: `url(${heroSrc})` }} />
        <div className="games-hero-scrim" aria-hidden="true" />

        {showDots ? (
          <div className="games-hero-dots" aria-hidden="true">
            {games.map((g, i) => (
              <button
                key={g.id}
                type="button"
                className={`games-hero-dot${i === index ? " is-active" : ""}`}
                onClick={() => goTo(i)}
                aria-label={g.title}
              />
            ))}
          </div>
        ) : null}

        <div className="games-hero-copy">
          <span className="games-hero-eyebrow">★ מועדפים · {countLabel} משחקים</span>
          <h2 className="games-hero-title">{current.title}</h2>
          {current.description ? <p className="games-hero-snippet">{current.description}</p> : null}
          <div className="games-hero-meta">
            {current.platform ? <span>{current.platform}</span> : null}
            {current.year ? <span>{current.year}</span> : null}
            {current.curated ? <span className="games-hero-pill">TOP</span> : null}
          </div>
          <button type="button" className="games-hero-play" onClick={() => onPlay(current)}>
            ▶ שחק עכשיו
          </button>
        </div>
      </div>

      <div className="games-hero-thumbs" role="tablist" aria-label="בחירת משחק מוצג">
        {games.map((g, i) => (
          <button
            key={g.id}
            type="button"
            role="tab"
            aria-selected={i === index}
            className={`games-hero-thumb${i === index ? " is-active" : ""}`}
            onClick={() => goTo(i)}
            title={g.title}
          >
            <img src={gameHeroImageUrl(g)} alt="" loading="lazy" referrerPolicy="no-referrer" />
            <span className="games-hero-thumb-label">{g.title}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
