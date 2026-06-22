const HF_TOKEN_STORAGE_KEY = "grovee-hf-token";

export function getHfToken(): string | undefined {
  const fromEnv = (import.meta.env.VITE_HF_TOKEN as string | undefined)?.trim();
  if (fromEnv) return fromEnv;
  if (typeof localStorage === "undefined") return undefined;
  const stored = localStorage.getItem(HF_TOKEN_STORAGE_KEY)?.trim();
  return stored || undefined;
}

export function setHfToken(token: string): void {
  if (typeof localStorage === "undefined") return;
  const t = token.trim();
  if (t) localStorage.setItem(HF_TOKEN_STORAGE_KEY, t);
  else localStorage.removeItem(HF_TOKEN_STORAGE_KEY);
}

export function getHfScannerBaseUrl(): string | undefined {
  const raw = (import.meta.env.VITE_HF_SCANNER_URL as string | undefined)?.trim();
  if (raw) return raw.replace(/\/$/, "");
  if (import.meta.env.DEV) return "/api/hf-scanner";
  return undefined;
}
