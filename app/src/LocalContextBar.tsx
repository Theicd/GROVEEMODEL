import { useEffect, useState } from "react";
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
};

export function LocalContextBar({ context }: Props) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  if (!context) return null;

  const tz = context.timezone;
  let timeLabel = "—";
  let dateLabel = "";
  try {
    timeLabel = new Intl.DateTimeFormat("he-IL", {
      timeZone: tz,
      hour: "2-digit",
      minute: "2-digit",
    }).format(now);
    dateLabel = new Intl.DateTimeFormat("he-IL", {
      timeZone: tz,
      weekday: "short",
      day: "numeric",
      month: "short",
    }).format(now);
  } catch {
    timeLabel = now.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
    dateLabel = now.toLocaleDateString("he-IL", { weekday: "short", day: "numeric", month: "short" });
  }

  const place = context.cityName ?? context.countryCode;
  const wx =
    context.localTempC != null
      ? `${WMO_ICON[context.localWeatherCode ?? 0] ?? "🌡"} ${Math.round(context.localTempC)}°`
      : null;

  return (
    <div className="local-context-bar" dir="ltr" title={`${context.timezone} · ${context.countryName}`}>
      <span className="local-context-time">{timeLabel}</span>
      <span className="local-context-sep" aria-hidden="true" />
      <span className="local-context-date">{dateLabel}</span>
      <span className="local-context-sep" aria-hidden="true" />
      <span className="local-context-place">{place}</span>
      {wx ? (
        <>
          <span className="local-context-sep" aria-hidden="true" />
          <span
            className={`local-context-weather${context.localTempC != null ? " local-context-weather--live" : ""}`}
          >
            {wx}
          </span>
        </>
      ) : null}
    </div>
  );
}
