import { useEffect, useMemo, useRef, useState } from "react";
import type { TimeWidgetData } from "./timeWidget/types";

type ClockParts = {
  hour: number;
  minute: number;
  second: number;
};

function readClockParts(date: Date, timezone: string): ClockParts {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    hour: "numeric",
    minute: "numeric",
    second: "numeric",
    hour12: false,
  });
  const parts = fmt.formatToParts(date);
  const pick = (type: Intl.DateTimeFormatPartTypes) =>
    Number(parts.find((p) => p.type === type)?.value ?? "0");
  let hour = pick("hour");
  if (hour === 24) hour = 0;
  return { hour, minute: pick("minute"), second: pick("second") };
}

function readSmoothSecond(date: Date, timezone: string): number {
  try {
    const fmt = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      fractionalSecondDigits: 3,
      hour12: false,
    } as Intl.DateTimeFormatOptions);
    const parts = fmt.formatToParts(date);
    const second = Number(parts.find((p) => p.type === "second")?.value ?? "0");
    const fracRaw = parts.find((p) => p.type === "fractionalSecond")?.value ?? "0";
    const frac = Number(fracRaw) / 10 ** fracRaw.length;
    return second + frac;
  } catch {
    const parts = readClockParts(date, timezone);
    return parts.second + date.getMilliseconds() / 1000;
  }
}

function formatDigital(date: Date, timezone: string) {
  return new Intl.DateTimeFormat("he-IL", {
    timeZone: timezone,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function formatDateLine(now: Date, timezone: string, utcOffsetLabel: string, dstActive?: boolean) {
  const weekdayDate = new Intl.DateTimeFormat("he-IL", {
    timeZone: timezone,
    weekday: "short",
    day: "numeric",
    month: "short",
  }).format(now);
  const offset = utcOffsetLabel.replace(/^UTC/i, "").trim();
  const offsetPart = offset ? `${offset}hrs` : "";
  const dstPart = dstActive ? "DST" : "";
  return [weekdayDate, offsetPart, dstPart].filter(Boolean).join(" · ");
}

function setHandAngle(el: SVGGElement | null, angle: number) {
  if (!el) return;
  el.setAttribute("transform", `rotate(${angle} 60 60)`);
}

function AnalogClockFace({ timezone, parts }: { timezone: string; parts: ClockParts }) {
  const hourRef = useRef<SVGGElement>(null);
  const minuteRef = useRef<SVGGElement>(null);
  const secondRef = useRef<SVGGElement>(null);

  const initialSecondAngle = parts.second * 6;
  const initialMinuteAngle = parts.minute * 6 + parts.second * 0.1;
  const initialHourAngle = (parts.hour % 12) * 30 + parts.minute * 0.5;

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const now = new Date();
      const live = readClockParts(now, timezone);
      const smoothSecond = readSmoothSecond(now, timezone);
      setHandAngle(hourRef.current, (live.hour % 12) * 30 + live.minute * 0.5);
      setHandAngle(minuteRef.current, live.minute * 6 + smoothSecond * 0.1);
      setHandAngle(secondRef.current, smoothSecond * 6);
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [timezone]);

  const numbers = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

  return (
    <div className="time-clock-analog" aria-hidden="true">
      <svg viewBox="0 0 120 120" className="time-clock-analog-svg" role="img">
        <circle cx="60" cy="60" r="54" className="time-clock-analog-face" />
        {numbers.map((n) => {
          const angle = ((n === 12 ? 0 : n) * 30 - 90) * (Math.PI / 180);
          const x = 60 + Math.cos(angle) * 42;
          const y = 60 + Math.sin(angle) * 42;
          return (
            <text key={n} x={x} y={y} className="time-clock-analog-num" textAnchor="middle" dominantBaseline="middle">
              {n}
            </text>
          );
        })}
        <g ref={hourRef} transform={`rotate(${initialHourAngle} 60 60)`} className="time-clock-hand time-clock-hand--hour">
          <line x1="60" y1="60" x2="60" y2="34" stroke="#f2f2f2" strokeWidth="3.5" strokeLinecap="round" />
        </g>
        <g ref={minuteRef} transform={`rotate(${initialMinuteAngle} 60 60)`} className="time-clock-hand time-clock-hand--minute">
          <line x1="60" y1="60" x2="60" y2="24" stroke="#f2f2f2" strokeWidth="2.5" strokeLinecap="round" />
        </g>
        <g ref={secondRef} transform={`rotate(${initialSecondAngle} 60 60)`} className="time-clock-hand time-clock-hand--second">
          <line x1="60" y1="60" x2="60" y2="18" stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round" />
        </g>
        <circle cx="60" cy="60" r="2.5" className="time-clock-hand-cap" />
      </svg>
    </div>
  );
}

export function TimeClockWidget({ data }: { data: TimeWidgetData }) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, [data.timezone]);

  const parts = useMemo(() => readClockParts(now, data.timezone), [now, data.timezone]);
  const digital = useMemo(() => formatDigital(now, data.timezone), [now, data.timezone]);
  const dateLine = useMemo(
    () => formatDateLine(now, data.timezone, data.utcOffsetLabel, data.dstActive),
    [now, data.timezone, data.utcOffsetLabel, data.dstActive],
  );

  return (
    <div className="time-clock-widget" dir="ltr">
      <div className="time-clock-widget-body">
        <div className="time-clock-widget-meta">
          <div className="time-clock-widget-digital">{digital}</div>
          <div className="time-clock-widget-place">{data.placeLabel}</div>
          <div className="time-clock-widget-date">{dateLine}</div>
        </div>
        <AnalogClockFace timezone={data.timezone} parts={parts} />
      </div>
    </div>
  );
}
