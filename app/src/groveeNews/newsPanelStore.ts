import type { NewsPanelPayload } from "./types";

let payload: NewsPanelPayload | null = null;
const listeners = new Set<(p: NewsPanelPayload | null) => void>();

export function setNewsPanelPayload(next: NewsPanelPayload | null): void {
  payload = next;
  listeners.forEach((cb) => cb(payload));
}

export function getNewsPanelPayload(): NewsPanelPayload | null {
  return payload;
}

export function subscribeNewsPanelPayload(cb: (p: NewsPanelPayload | null) => void): () => void {
  listeners.add(cb);
  cb(payload);
  return () => listeners.delete(cb);
}

export function clearNewsPanelPayload(): void {
  setNewsPanelPayload(null);
}
