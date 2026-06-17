import { GROVEE_INFO_CARDS } from "../groveeInfoContent";
import { GroveeInfoIllustration } from "./GroveeInfoIllustrations";

type GroveeInfoModalProps = {
  open: boolean;
  onClose: () => void;
};

export function GroveeInfoModal({ open, onClose }: GroveeInfoModalProps) {
  if (!open) return null;

  return (
    <div
      className="grovee-info-page"
      role="dialog"
      aria-modal="true"
      aria-labelledby="info-modal-title"
    >
      <div className="grovee-info-page__scanlines" aria-hidden="true" />

      <header className="grovee-info-page__top">
        <div className="grovee-info-page__brand" dir="ltr">
          <span className="grovee-info-page__mark">G</span>
          <span className="grovee-info-page__name">GROVEE</span>
        </div>

        <div className="grovee-info-page__headline">
          <p className="grovee-info-page__eyebrow" dir="ltr">
            LOCAL AI · BROWSER
          </p>
          <h1 id="info-modal-title" className="grovee-info-page__title">
            איך זה עובד?
          </h1>
        </div>

        <button
          type="button"
          className="btn-hal btn-hal--small grovee-info-page__back"
          onClick={onClose}
        >
          <span className="btn-hal__shine" aria-hidden="true" />
          חזרה
        </button>
      </header>

      <main className="grovee-info-page__grid" aria-label="מידע על הממשק">
        {GROVEE_INFO_CARDS.map((card) => (
          <article key={card.id} className="grovee-info-card" data-card={card.id}>
            <h2 className="grovee-info-card__title">{card.title}</h2>
            <GroveeInfoIllustration cardId={card.id} />
            <p className="grovee-info-card__text">{card.body}</p>
            {card.tags?.length ? (
              <ul className="grovee-info-card__tags" aria-label="תגיות">
                {card.tags.map((tag) => (
                  <li key={tag}>{tag}</li>
                ))}
              </ul>
            ) : null}
            {card.links?.length ? (
              <p className="grovee-info-card__links">
                {card.links.map((link, i) => (
                  <span key={link.href}>
                    {i > 0 ? " · " : null}
                    <a href={link.href} target="_blank" rel="noreferrer">
                      {link.label}
                    </a>
                  </span>
                ))}
              </p>
            ) : null}
          </article>
        ))}
      </main>

      <footer className="grovee-info-page__foot" dir="ltr">
        <span>GROVEE · GEMMA 4 E2B · Transformers.js · מקומי בדפדפן</span>
      </footer>
    </div>
  );
}
