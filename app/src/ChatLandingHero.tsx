import { useMemo } from "react";
import { LANDING_HEADLINES, LANDING_SUGGESTION_SETS, type LandingSuggestion } from "./chatLandingContent";

export type LandingContent = {
  headline: string;
  suggestions: LandingSuggestion[];
};

function pickLandingContent(): LandingContent {
  return {
    headline: LANDING_HEADLINES[Math.floor(Math.random() * LANDING_HEADLINES.length)],
    suggestions: LANDING_SUGGESTION_SETS[Math.floor(Math.random() * LANDING_SUGGESTION_SETS.length)],
  };
}

export function useLandingContent(): LandingContent {
  return useMemo(() => pickLandingContent(), []);
}

export function ChatLandingHeadline({ text }: { text: string }) {
  return (
    <div className="chat-landing-headline-wrap" dir="ltr">
      <h1 className="chat-landing-headline">{text}</h1>
    </div>
  );
}

export function ChatLandingSuggestions({
  suggestions,
  onSuggestionClick,
}: {
  suggestions: LandingSuggestion[];
  onSuggestionClick: (prompt: string) => void;
}) {
  return (
    <div className="chat-landing-suggestions" dir="ltr">
      {suggestions.map((item) => (
        <button
          key={item.label}
          type="button"
          className="chat-landing-chip"
          onClick={() => onSuggestionClick(item.prompt)}
          title={item.label}
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
