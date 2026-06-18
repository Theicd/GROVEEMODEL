import type { SearchResultsPayload } from "./types";

let payload: SearchResultsPayload | null = null;
const listeners = new Set<(p: SearchResultsPayload | null) => void>();

export function setSearchResultsPayload(next: SearchResultsPayload | null): void {
  payload = next;
  listeners.forEach((cb) => cb(payload));
}

export function getSearchResultsPayload(): SearchResultsPayload | null {
  return payload;
}

export function subscribeSearchResultsPayload(
  cb: (p: SearchResultsPayload | null) => void,
): () => void {
  listeners.add(cb);
  cb(payload);
  return () => listeners.delete(cb);
}

export function clearSearchResultsPayload(): void {
  setSearchResultsPayload(null);
}
