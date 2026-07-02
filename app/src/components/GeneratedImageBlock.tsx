type Props = {
  url: string;
  prompt?: string;
  uiLang?: "he" | "en";
};

export function GeneratedImageBlock({ url, prompt, uiLang = "he" }: Props) {
  const label = uiLang === "he" ? "תמונה שנוצרה" : "Generated image";
  return (
    <div className="generated-image-block" data-testid="generated-image-block">
      <img
        src={url}
        alt={prompt?.slice(0, 120) || label}
        className="generated-image-block__img"
        loading="lazy"
      />
      {prompt ? (
        <p className="generated-image-block__caption" dir="auto">
          {prompt.slice(0, 200)}
        </p>
      ) : null}
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="generated-image-block__link"
      >
        {uiLang === "he" ? "פתח בגודל מלא" : "Open full size"}
      </a>
    </div>
  );
}
