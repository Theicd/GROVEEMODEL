/** In-app audit trail of model requests/responses for debugging. */

export type ModelActivityDirection = "out" | "in" | "system";

export type ModelActivityEntry = {
  id: string;
  ts: number;
  direction: ModelActivityDirection;
  kind: string;
  title: string;
  detail: string;
  meta?: Record<string, string | number | boolean>;
};

const MAX_ENTRIES = 120;

export const newActivityId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `act-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

export const createModelActivityEntry = (
  partial: Omit<ModelActivityEntry, "id" | "ts"> & { id?: string; ts?: number },
): ModelActivityEntry => ({
  id: partial.id ?? newActivityId(),
  ts: partial.ts ?? Date.now(),
  direction: partial.direction,
  kind: partial.kind,
  title: partial.title,
  detail: partial.detail,
  meta: partial.meta,
});

export const appendModelActivity = (
  entries: ModelActivityEntry[],
  entry: Omit<ModelActivityEntry, "id" | "ts"> & { id?: string; ts?: number },
): ModelActivityEntry[] => {
  const next = [createModelActivityEntry(entry), ...entries];
  return next.length > MAX_ENTRIES ? next.slice(0, MAX_ENTRIES) : next;
};

export const formatActivityTime = (ts: number): string => {
  const d = new Date(ts);
  return d.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};

export const directionLabel = (d: ModelActivityDirection): string => {
  if (d === "out") return "→ מודל";
  if (d === "in") return "← מודל";
  return "⚙ מערכת";
};

const formatActivityDateTime = (ts: number): string => {
  const d = new Date(ts);
  return d.toLocaleString("he-IL", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

/** Export full activity log as plain text (oldest → newest). */
export const formatActivityLogForCopy = (entries: ModelActivityEntry[]): string => {
  const sorted = [...entries].sort((a, b) => a.ts - b.ts);
  const lines: string[] = [
    "GROVEE — Model Activity Log",
    `Exported: ${formatActivityDateTime(Date.now())}`,
    `Entries: ${sorted.length}`,
    "",
  ];

  sorted.forEach((entry, index) => {
    lines.push("═".repeat(72));
    lines.push(`#${index + 1}  ${formatActivityDateTime(entry.ts)}  |  ${directionLabel(entry.direction)}  |  ${entry.kind}`);
    lines.push(`Title: ${entry.title}`);
    if (entry.meta && Object.keys(entry.meta).length > 0) {
      lines.push("Meta:");
      for (const [k, v] of Object.entries(entry.meta)) {
        lines.push(`  ${k}: ${String(v)}`);
      }
    }
    lines.push("─".repeat(72));
    lines.push(entry.detail.trim() || "(empty)");
    lines.push("");
  });

  return lines.join("\n").trimEnd();
};
