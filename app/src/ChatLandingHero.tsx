import { useEffect, useState } from "react";
import {
  LANDING_ROTATION_MS,
  pickLandingHeadline,
  pickRotatingLandingSuggestions,
  type LandingSuggestion,
} from "./chatLandingContent";

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
            {item.icon}
          </span>
          <span className="chat-landing-chip-label">{item.label}</span>
        </button>
      ))}
    </div>
  );
}
