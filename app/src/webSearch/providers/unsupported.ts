import type { SearchSourceResult } from "../types";

/** Graceful stub when a source needs API key or server proxy. */
export const fetchUnsupportedSource = (
  provider: SearchSourceResult["provider"],
  label: string,
  reasonHe: string,
): SearchSourceResult => ({
  provider,
  label,
  ok: false,
  text: "",
  error: reasonHe,
  latencyMs: 0,
});
