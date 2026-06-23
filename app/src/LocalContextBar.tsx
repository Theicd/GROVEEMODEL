import { useEffect, useState } from "react";
import { NetworkStatusIcon } from "./components/NetworkStatusIcon";
import { useNetworkStatus } from "./hooks/useNetworkStatus";
import type { StartupContext } from "./startupContext/types";

const WMO_ICON: Record<number, string> = {
  0: "☀️",
  1: "🌤",
  2: "⛅",
  3: "☁️",
  45: "🌫",
  48: "🌫",
  51: "🌦",
  61: "🌧",
  63: "🌧",
  65: "🌧",
  71: "🌨",
  80: "🌦",
  95: "⛈",
};

type Props = {
  context: StartupContext | null;
  uiLang?: "he" | "en";
  /** Compact row for mobile chat header: network icon + time + date only */
  variant?: "full" | "header";
  className?: string;
};

function browserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Jerusalem";
  } catch {
    return "Asia/Jerusalem";
  }
}

export function LocalContextBar({ context, uiLang = "he", variant = "full", className }: Props) {
  const isHeader = variant === "header";
  const networkStatus = useNetworkStatus();
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  const tz = context?.timezone ?? browserTimezone();
  let timeLabel = "—";
  let dateLabel = "";
  try {
    timeLabel = new Intl.DateTimeFormat(uiLang === "he" ? "he-IL" : "en-US", {
      timeZone: tz,
      hour: "2-digit",
      minute: "2-digit",
    }).format(now);
    dateLabel = new Intl.DateTimeFormat(uiLang === "he" ? "he-IL" : "en-US", {
      timeZone: tz,
      weekday: "short",
      day: "numeric",
      month: "short",
    }).format(now);
  } catch {
    timeLabel = now.toLocaleTimeString(uiLang === "he" ? "he-IL" : "en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
    dateLabel = now.toLocaleDateString(uiLang === "he" ? "he-IL" : "en-US", {
      weekday: "short",
      day: "numeric",
      month: "short",
    });
  }

  const place = context?.cityName ?? context?.countryCode ?? (uiLang === "he" ? "מקומי" : "Local");
  const wx =
    context?.localTempC != null
      ? `${WMO_ICON[context.localWeatherCode ?? 0] ?? "🌡"} ${Math.round(context.localTempC)}°`
      : null;

  return (
    <div
      className={`local-context-bar${isHeader ? " local-context-bar--header" : ""}${className ? ` ${className}` : ""}`}
      dir="ltr"
      title={context ? `${context.timezone} · ${context.countryName}` : tz}
    >
      <NetworkStatusIcon status={networkStatus} uiLang={uiLang} iconOnly={isHeader} />
      <span className="local-context-sep" aria-hidden="true" />
      <span className="local-context-time">{timeLabel}</span>
      <span className="local-context-sep" aria-hidden="true" />
      <span className="local-context-date">{dateLabel}</span>
      {!isHeader ? (
        <>
          <span className="local-context-sep" aria-hidden="true" />
          <span className="local-context-place">{place}</span>
          {wx ? (
            <>
              <span className="local-context-sep" aria-hidden="true" />
              <span
                className={`local-context-weather${context?.localTempC != null ? " local-context-weather--live" : ""}`}
              >
                {wx}
              </span>
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
