// @ts-nocheck
const STORAGE_KEY = "gn-user-news-profile";

export type UserNewsProfile = {
  locale: string;
  uiLanguage: string;
  pollTier: "core" | "full";
};

const DEFAULT_PROFILE: UserNewsProfile = {
  locale: "en-US",
  uiLanguage: "en",
  pollTier: "core",
};

const listeners = new Set<() => void>();

function notify(): void {
  listeners.forEach((cb) => cb());
}

/** Languages with RTL layout for the world feed. */
export const RTL_LANGUAGES = new Set(["he", "ar", "fa", "ur"]);

export const UI_LANGUAGE_OPTIONS = [
  { code: "en", label: "English", native: "English" },
  { code: "he", label: "Hebrew", native: "עברית" },
  { code: "fr", label: "French", native: "Français" },
  { code: "de", label: "German", native: "Deutsch" },
  { code: "es", label: "Spanish", native: "Español" },
  { code: "pt", label: "Portuguese", native: "Português" },
  { code: "ru", label: "Russian", native: "Русский" },
  { code: "ja", label: "Japanese", native: "日本語" },
  { code: "zh", label: "Chinese", native: "中文" },
  { code: "ko", label: "Korean", native: "한국어" },
  { code: "ar", label: "Arabic", native: "العربية" },
] as const;

export type UiLanguageCode = (typeof UI_LANGUAGE_OPTIONS)[number]["code"];

const LOCALE_BY_LANG: Record<string, string> = {
  en: "en-US",
  he: "he-IL",
  ru: "ru-RU",
  fr: "fr-FR",
  de: "de-DE",
  es: "es-ES",
  pt: "pt-BR",
  ja: "ja-JP",
  zh: "zh-CN",
  ko: "ko-KR",
  ar: "ar-SA",
};

export function localeForLanguage(code: string): string {
  return LOCALE_BY_LANG[code] ?? `${code}-${code.toUpperCase()}`;
}

export function isRtlLanguage(code: string): boolean {
  return RTL_LANGUAGES.has(code);
}

export function detectDefaultProfile(): UserNewsProfile {
  if (typeof navigator === "undefined") return DEFAULT_PROFILE;
  const locale = navigator.language || "en-US";
  const ui = locale.split("-")[0].toLowerCase();
  const supported = UI_LANGUAGE_OPTIONS.some((o) => o.code === ui);
  return {
    locale: supported ? localeForLanguage(ui) : DEFAULT_PROFILE.locale,
    uiLanguage: supported ? ui : "en",
    pollTier: "core",
  };
}

export function getUserNewsProfile(): UserNewsProfile {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return detectDefaultProfile();
    const parsed = JSON.parse(raw) as Partial<UserNewsProfile> & { regions?: string[] };
    const uiLanguage = parsed.uiLanguage ?? DEFAULT_PROFILE.uiLanguage;
    const validLang = UI_LANGUAGE_OPTIONS.some((o) => o.code === uiLanguage);
    return {
      locale: parsed.locale ?? localeForLanguage(validLang ? uiLanguage : "en"),
      uiLanguage: validLang ? uiLanguage : "en",
      pollTier: parsed.pollTier ?? "core",
    };
  } catch {
    return detectDefaultProfile();
  }
}

export function setUserNewsProfile(patch: Partial<UserNewsProfile>): UserNewsProfile {
  const next = { ...getUserNewsProfile(), ...patch };
  if (patch.uiLanguage) {
    next.locale = localeForLanguage(patch.uiLanguage);
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
  notify();
  return next;
}

export function subscribeUserNewsProfile(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
