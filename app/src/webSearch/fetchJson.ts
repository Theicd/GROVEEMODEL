import type { FetchJsonOptions } from "./types";
import { defaultFetchTimeoutMs, proxyAwareFetch } from "./proxyFetch";

export class FetchTimeoutError extends Error {
  constructor(url: string) {
    super(`Timeout fetching ${url}`);
    this.name = "FetchTimeoutError";
  }
}

export async function fetchJson<T>(
  url: string,
  init?: RequestInit,
  options: FetchJsonOptions = {},
): Promise<T> {
  const timeoutMs = options.timeoutMs ?? defaultFetchTimeoutMs();
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await proxyAwareFetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...options.headers,
        ...init?.headers,
      },
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} for ${url}`);
    }
    return (await response.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new FetchTimeoutError(url);
    }
    throw err;
  } finally {
    globalThis.clearTimeout(timer);
  }
}

export async function fetchText(
  url: string,
  init?: RequestInit,
  options: FetchJsonOptions = {},
): Promise<string> {
  const timeoutMs = options.timeoutMs ?? defaultFetchTimeoutMs();
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await proxyAwareFetch(url, { ...init, signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.text();
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new FetchTimeoutError(url);
    }
    throw err;
  } finally {
    globalThis.clearTimeout(timer);
  }
}
