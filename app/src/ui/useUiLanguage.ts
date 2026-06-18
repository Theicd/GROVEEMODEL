import { useEffect, useState } from "react";
import {
  getUserNewsProfile,
  isRtlLanguage,
  setUserNewsProfile,
  subscribeUserNewsProfile,
  type UiLanguageCode,
} from "../groveeNews/engine/settings/userNewsProfile";

export type ChatUiLanguage = "he" | "en";

export function applyUiLanguageToDocument(lang: string): void {
  if (typeof document === "undefined") return;
  document.documentElement.lang = lang;
  document.documentElement.dir = isRtlLanguage(lang) ? "rtl" : "ltr";
}

export function getChatUiLanguage(): ChatUiLanguage {
  const code = getUserNewsProfile().uiLanguage;
  return code === "he" ? "he" : "en";
}

export function setChatUiLanguage(lang: ChatUiLanguage): void {
  setUserNewsProfile({ uiLanguage: lang });
  applyUiLanguageToDocument(lang);
}

export function useUiLanguage(): ChatUiLanguage {
  const [lang, setLang] = useState<ChatUiLanguage>(() => getChatUiLanguage());

  useEffect(() => {
    applyUiLanguageToDocument(getUserNewsProfile().uiLanguage);
    return subscribeUserNewsProfile(() => {
      setLang(getChatUiLanguage());
      applyUiLanguageToDocument(getUserNewsProfile().uiLanguage);
    });
  }, []);

  return lang;
}

export function uiLanguageCode(lang: ChatUiLanguage): UiLanguageCode {
  return lang;
}
