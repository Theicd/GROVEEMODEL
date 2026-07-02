import { useEffect, useRef, useState } from "react";
import type { WeatherIconKind, WeatherWidgetData } from "../weatherWidget/types";
import "../weatherWidget/weatherWidget.css";

type Props = {
  data: WeatherWidgetData;
  uiLang?: "he" | "en";
};

function WeatherMainIcon({ kind }: { kind: WeatherIconKind }) {
  if (kind === "clear" || kind === "partly-cloudy") {
    return (
      <div className="weather-widget-card__icon-wrap" aria-hidden="true">
        <div className="weather-widget-card__sun" />
        <svg className="weather-widget-card__cloud" viewBox="0 0 50 28">
          <path d="M12 26H38a10 10 0 0 0 2.5-19.7A12 12 0 0 0 18 4a10 10 0 0 0-6 22z" fill="#c8cdd5" />
        </svg>
      </div>
    );
  }
  if (kind === "rain" || kind === "drizzle" || kind === "thunder") {
    return (
      <div className="weather-widget-card__icon-wrap" aria-hidden="true">
        <svg width="56" height="56" viewBox="0 0 56 56">
          <path d="M10 34h36a8 8 0 0 0 0-16 10 10 0 0 0-19.2-3.2A8 8 0 0 0 10 34z" fill="#8a8f9a" />
          <line x1="18" y1="38" x2="16" y2="46" stroke="#5b9bd5" strokeWidth="2" strokeLinecap="round" />
          <line x1="28" y1="38" x2="26" y2="48" stroke="#5b9bd5" strokeWidth="2" strokeLinecap="round" />
          <line x1="38" y1="38" x2="36" y2="45" stroke="#5b9bd5" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </div>
    );
  }
  if (kind === "snow") {
    return (
      <div className="weather-widget-card__icon-wrap" aria-hidden="true">
        <svg width="56" height="56" viewBox="0 0 56 56">
          <path d="M10 34h36a8 8 0 0 0 0-16 10 10 0 0 0-19.2-3.2A8 8 0 0 0 10 34z" fill="#9ca0aa" />
          <text x="20" y="50" fill="#c8cdd5" fontSize="10">
            ❄
          </text>
        </svg>
      </div>
    );
  }
  return (
    <div className="weather-widget-card__icon-wrap" aria-hidden="true">
      <svg width="56" height="56" viewBox="0 0 56 56">
        <path d="M8 36h40a10 10 0 0 0 0-20 12 12 0 0 0-23-4A10 10 0 0 0 8 36z" fill="#8a8f9a" />
      </svg>
    </div>
  );
}

function rainClass(pct?: number): string {
  if (pct == null) return "weather-widget-card__rain weather-widget-card__rain--low";
  if (pct >= 40) return "weather-widget-card__rain weather-widget-card__rain--high";
  if (pct >= 15) return "weather-widget-card__rain weather-widget-card__rain--mid";
  return "weather-widget-card__rain weather-widget-card__rain--low";
}

function useAnimatedTemp(target: number, durationMs = 1200) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const isFloat = target % 1 !== 0;
    let frame = 0;
    const tick = (now: number) => {
      const progress = Math.min((now - start) / durationMs, 1);
      const eased = progress === 1 ? 1 : 1 - 2 ** (-10 * progress);
      const current = eased * target;
      setDisplay(isFloat ? Number(current.toFixed(1)) : Math.round(current));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, durationMs]);
  return display;
}

export function WeatherWidgetCard({ data, uiLang = "he" }: Props) {
  const tempDisplay = useAnimatedTemp(data.temperatureC);
  const barRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const t = window.setTimeout(() => {
      data.forecast.forEach((day, i) => {
        const el = barRefs.current[i];
        if (el) el.style.width = `${day.tempBarPct}%`;
      });
    }, 700);
    return () => window.clearTimeout(t);
  }, [data.forecast]);

  const humidityLabel = uiLang === "he" ? "לחות" : "Humidity";
  const windLabel = uiLang === "he" ? "רוח" : "Wind";
  const compact = data.forecast.length === 0;
  const forecastTitle =
    uiLang === "he"
      ? `תחזית ${data.forecast.length} ימים`
      : `${data.forecast.length}-day forecast`;
  const liveLabel = uiLang === "he" ? "עדכני" : "Live";

  return (
    <div
      className={`weather-widget-card${compact ? " weather-widget-card--compact" : ""}`}
      role="region"
      aria-label={`${data.cityName} — ${data.condition}`}
      data-testid="weather-widget-card"
    >
      <div className="weather-widget-card__header-bg" />
      <div className="weather-widget-card__top">
        <div>
          <div className="weather-widget-card__location-name">
            <svg
              className="weather-widget-card__pin"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
              <circle cx="12" cy="10" r="3" />
            </svg>
            {data.cityName}
          </div>
          {data.regionLabel ? (
            <div className="weather-widget-card__region">{data.regionLabel}</div>
          ) : null}
        </div>
        <WeatherMainIcon kind={data.iconKind} />
      </div>

      <div className="weather-widget-card__temp-row">
        <div className="weather-widget-card__temp" aria-label={`${data.temperatureC}°C`}>
          {tempDisplay}
          <span className="weather-widget-card__temp-unit">°C</span>
        </div>
        <div className="weather-widget-card__badge">
          <span className="weather-widget-card__badge-dot" />
          {data.condition}
        </div>
      </div>

      <div className="weather-widget-card__metrics">
        {data.humidityPct != null ? (
          <div className="weather-widget-card__metric">
            <div className="weather-widget-card__metric-icon weather-widget-card__metric-icon--humidity">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
              </svg>
            </div>
            <div>
              <div className="weather-widget-card__metric-label">{humidityLabel}</div>
              <div className="weather-widget-card__metric-value">{Math.round(data.humidityPct)}%</div>
            </div>
          </div>
        ) : null}
        {data.windKmh != null ? (
          <div className="weather-widget-card__metric">
            <div className="weather-widget-card__metric-icon weather-widget-card__metric-icon--wind">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2" />
              </svg>
            </div>
            <div>
              <div className="weather-widget-card__metric-label">{windLabel}</div>
              <div className="weather-widget-card__metric-value">{data.windKmh.toFixed(1)} km/h</div>
              {data.windDirectionDeg != null && !compact ? (
                <div className="weather-widget-card__metric-sub">
                  {uiLang === "he" ? "כיוון" : "dir"} {Math.round(data.windDirectionDeg)}°
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>

      {data.forecast.length ? (
        <>
          <div className="weather-widget-card__divider" />
          <section className="weather-widget-card__forecast">
            <div className="weather-widget-card__forecast-title">{forecastTitle}</div>
            <div className="weather-widget-card__forecast-list">
              {data.forecast.map((day, i) => (
                <div key={day.dateIso} className="weather-widget-card__forecast-item">
                  <span className="weather-widget-card__forecast-day">{day.dayLabel}</span>
                  <div className="weather-widget-card__temp-bar-wrap">
                    <div
                      ref={(el) => {
                        barRefs.current[i] = el;
                      }}
                      className="weather-widget-card__temp-bar"
                    />
                  </div>
                  <div className="weather-widget-card__forecast-temps">
                    <span className="weather-widget-card__temp-high">{day.maxC.toFixed(1)}°</span>
                    <span className="weather-widget-card__temp-low">/ {day.minC.toFixed(1)}°</span>
                  </div>
                  {day.precipChancePct != null ? (
                    <span className={rainClass(day.precipChancePct)}>{day.precipChancePct}%</span>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        </>
      ) : null}

      <footer className="weather-widget-card__footer">
        <span>{data.sourceLabel}</span>
        <span className="weather-widget-card__live">
          <span className="weather-widget-card__live-dot" />
          {liveLabel}
        </span>
      </footer>
    </div>
  );
}
