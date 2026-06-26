import type { EpgProgram } from "./types";

export const EPG_GRID_SLOT_MIN = 30;
export const EPG_GRID_HOURS = 4;
export const EPG_GRID_SLOT_PX = 128;
export const EPG_GRID_ROW_PX = 52;

export type EpgGridWindow = {
  start: Date;
  end: Date;
  slots: Date[];
  totalPx: number;
};

export type EpgGridBlock = {
  program: EpgProgram;
  leftPx: number;
  widthPx: number;
  live: boolean;
};

export function buildEpgGridWindow(now = new Date()): EpgGridWindow {
  const start = new Date(now);
  start.setSeconds(0, 0);
  start.setMinutes(Math.floor(start.getMinutes() / EPG_GRID_SLOT_MIN) * EPG_GRID_SLOT_MIN);
  const end = new Date(start.getTime() + EPG_GRID_HOURS * 60 * 60 * 1000);
  const slots: Date[] = [];
  for (let t = start.getTime(); t < end.getTime(); t += EPG_GRID_SLOT_MIN * 60_000) {
    slots.push(new Date(t));
  }
  return { start, end, slots, totalPx: slots.length * EPG_GRID_SLOT_PX };
}

export function layoutProgramsInWindow(programs: EpgProgram[], window: EpgGridWindow, now = new Date()): EpgGridBlock[] {
  const span = window.end.getTime() - window.start.getTime();
  if (span <= 0) return [];

  return programs
    .filter((p) => p.end.getTime() > window.start.getTime() && p.start.getTime() < window.end.getTime())
    .map((program) => {
      const visStart = Math.max(program.start.getTime(), window.start.getTime());
      const visEnd = Math.min(program.end.getTime(), window.end.getTime());
      const leftPx = ((visStart - window.start.getTime()) / span) * window.totalPx;
      const widthPx = Math.max(((visEnd - visStart) / span) * window.totalPx, 48);
      const live = program.start <= now && program.end > now;
      return { program, leftPx, widthPx, live };
    });
}

export function formatGridTime(d: Date, rtl: boolean): string {
  return d.toLocaleTimeString(rtl ? "he-IL" : "en-US", { hour: "numeric", minute: "2-digit", hour12: true });
}

export function formatGridRange(start: Date, end: Date, rtl: boolean): string {
  return `${formatGridTime(start, rtl)} – ${formatGridTime(end, rtl)}`;
}
