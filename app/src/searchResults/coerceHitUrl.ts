/** Normalize provider URLs — GDACS sometimes returns objects instead of strings. */
export const coerceHttpUrl = (raw: unknown, fallback: string): string => {
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    return trimmed.startsWith("http") ? trimmed : fallback;
  }
  if (raw && typeof raw === "object") {
    const obj = raw as Record<string, unknown>;
    for (const key of ["url", "href", "report", "link", "details"]) {
      const nested = coerceHttpUrl(obj[key], "");
      if (nested) return nested;
    }
    for (const value of Object.values(obj)) {
      const nested = coerceHttpUrl(value, "");
      if (nested) return nested;
    }
  }
  return fallback;
};

export const coerceText = (raw: unknown, fallback = ""): string => {
  if (raw == null) return fallback;
  if (typeof raw === "string") return raw.trim() || fallback;
  if (typeof raw === "number" || typeof raw === "boolean") return String(raw);
  return fallback;
};
