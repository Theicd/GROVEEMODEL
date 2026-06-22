import type { OnlineGame } from "./gameSearch/types";

import { formatPopularityLabel } from "./gameSearch/gameAliases";



type Props = {
  game: OnlineGame;
  compact?: boolean;
  onPlay: (game: OnlineGame) => void;
  isFavorite?: boolean;
  onToggleFavorite?: (game: OnlineGame) => void;
};



function ratingStars(rating: number | null | undefined): string | null {

  if (!rating || rating <= 0) return null;

  const full = Math.round(rating);

  return "★".repeat(Math.min(5, Math.max(1, full)));

}



export function GameCard({ game, compact, onPlay, isFavorite, onToggleFavorite }: Props) {

  const stars = ratingStars(game.rating);

  const popularity = formatPopularityLabel(game.downloads);

  const platformLabel = game.platform || "Browser";

  const yearLabel = game.year ? String(game.year) : null;



  return (

    <article

      className={`game-card${compact ? " game-card--compact" : ""}${game.curated ? " game-card--curated" : ""}`}

      onClick={() => onPlay(game)}

      onKeyDown={(e) => {

        if (e.key === "Enter" || e.key === " ") {

          e.preventDefault();

          onPlay(game);

        }

      }}

      role="button"

      tabIndex={0}

      aria-label={`שחק ${game.title}`}

    >

      <div className="game-card-thumb-wrap">

        <img

          className="game-card-thumb"

          src={game.thumbnail}

          alt=""

          loading="lazy"

          onError={(e) => {

            (e.target as HTMLImageElement).style.opacity = "0.35";

          }}

        />

        <span className="game-card-badge">{platformLabel.toUpperCase()}</span>

        {game.curated ? <span className="game-card-curated">TOP</span> : null}
        {onToggleFavorite ? (
          <button
            type="button"
            className={`game-card-fav${isFavorite ? " is-active" : ""}`}
            aria-label={isFavorite ? "הסר ממועדפים" : "הוסף למועדפים"}
            title={isFavorite ? "הסר ממועדפים" : "הוסף למועדפים"}
            onClick={(e) => {
              e.stopPropagation();
              onToggleFavorite(game);
            }}
          >
            {isFavorite ? "★" : "☆"}
          </button>
        ) : null}
      </div>

      <div className="game-card-body">

        <h3 className="game-card-title">{game.title}</h3>

        <div className="game-card-meta">

          {yearLabel ? <span>{yearLabel}</span> : null}

          {stars ? (

            <span className="game-card-rating" title={`${game.rating?.toFixed(1)} / 5`}>

              {stars}

              {game.reviewsCount ? ` (${game.reviewsCount})` : ""}

            </span>

          ) : null}

          {popularity ? <span className="game-card-pop">{popularity}</span> : null}

        </div>

        {!compact && game.description ? (

          <p className="game-card-desc">{game.description}</p>

        ) : null}

        <button

          type="button"

          className="game-card-play-btn"

          onClick={(e) => {

            e.stopPropagation();

            onPlay(game);

          }}

        >

          ▶ שחק עכשיו

        </button>

      </div>

    </article>

  );

}

