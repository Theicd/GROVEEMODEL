import { useEffect, useState } from "react";
import {
  LANDING_ROTATION_MS,
  pickLandingHeadline,
  pickRotatingLandingSuggestions,
  type LandingCategory,
  type LandingSuggestion,
} from "./chatLandingContent";

function LandingChipIcon({ category }: { category: LandingCategory }) {
  const common = {
    width: 18,
    height: 18,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  switch (category) {
    case "write":
    case "rewrite":
      return (
        <svg {...common}>
          <path d="M12 20h9" />
          <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
        </svg>
      );
    case "code":
    case "games":
      return (
        <svg {...common}>
          <polyline points="16 18 22 12 16 6" />
          <polyline points="8 6 2 12 8 18" />
        </svg>
      );
    case "image":
      return (
        <svg {...common}>
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <path d="M21 15l-5-5L5 21" />
        </svg>
      );
    case "globe":
    case "translate":
    case "search":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="10" />
          <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
          <path d="M2 12h20" />
        </svg>
      );
    case "camera":
      return (
        <svg {...common}>
          <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
          <circle cx="12" cy="13" r="3" />
        </svg>
      );
    case "think":
    case "explain":
    case "ideas":
      return (
        <svg {...common}>
          <path d="M9 18h6" />
          <path d="M10 22h4" />
          <path d="M12 2a7 7 0 0 0-4 12v2h8v-2a7 7 0 0 0-4-12z" />
        </svg>
      );
    case "summarize":
    case "plan":
    case "learn":
      return (
        <svg {...common}>
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
        </svg>
      );
  }

  return (
    <svg {...common}>
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

export type LandingContent = {
  headline: string;
  suggestions: LandingSuggestion[];
  rotationKey: number;
};

function pickLandingContent(): LandingContent {
  return {
    headline: pickLandingHeadline(),
    suggestions: pickRotatingLandingSuggestions(3),
    rotationKey: 0,
  };
}

export function useLandingContent(rotateMs = LANDING_ROTATION_MS): LandingContent {
  const [content, setContent] = useState(pickLandingContent);

  useEffect(() => {
    const id = window.setInterval(() => {
      setContent((prev) => ({
        headline: prev.headline,
        suggestions: pickRotatingLandingSuggestions(3),
        rotationKey: prev.rotationKey + 1,
      }));
    }, rotateMs);
    return () => window.clearInterval(id);
  }, [rotateMs]);

  return content;
}

export function ChatLandingHeadline({ text }: { text: string }) {
  return (
    <div className="chat-landing-headline-wrap" dir="auto">
      <h1 className="chat-landing-headline">{text}</h1>
    </div>
  );
}

export function ChatLandingSuggestions({
  suggestions,
  rotationKey,
  onSuggestionClick,
}: {
  suggestions: LandingSuggestion[];
  rotationKey?: number;
  onSuggestionClick: (prompt: string) => void;
}) {
  return (
    <div
      className="chat-landing-suggestions"
      dir="auto"
      key={rotationKey ?? 0}
      aria-live="polite"
      aria-atomic="true"
    >
      {suggestions.map((item) => (
        <button
          key={`${item.category}-${item.label}-${item.prompt}`}
          type="button"
          className="chat-landing-chip"
          onClick={() => onSuggestionClick(item.prompt)}
          title={item.prompt}
        >
          <span className="chat-landing-chip-icon" aria-hidden="true">
            <LandingChipIcon category={item.category} />
          </span>
          <span className="chat-landing-chip-label">{item.label}</span>
        </button>
      ))}
    </div>
  );
}
