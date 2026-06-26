import { translateTexts } from "../../groveeNews/engine/translate/googleTranslate";
import type { ChatUiLanguage } from "../../ui/useUiLanguage";
import { needsHebrewTranslation } from "../../tmdb/tmdbLocalize";

export type LocalizableEpgCopy = {
  title: string;
  description?: string | null;
  subTitle?: string | null;
};

/** Translate EPG programme description for Hebrew UI — never titles (idioms break). */
export async function localizeEpgCopyForUi(
  copy: LocalizableEpgCopy,
  uiLang: ChatUiLanguage,
): Promise<LocalizableEpgCopy> {
  if (uiLang !== "he") return copy;

  const pending: Array<"description" | "subTitle"> = [];
  const texts: string[] = [];
  if (needsHebrewTranslation(copy.description)) {
    pending.push("description");
    texts.push(copy.description!);
  } else if (needsHebrewTranslation(copy.subTitle)) {
    pending.push("subTitle");
    texts.push(copy.subTitle!);
  }
  if (!texts.length) return copy;

  try {
    const { texts: translated } = await translateTexts(texts, "he", "en");
    let idx = 0;
    let description = copy.description;
    let subTitle = copy.subTitle;
    for (const field of pending) {
      const value = translated[idx++]?.trim();
      if (!value) continue;
      if (field === "description") description = value;
      else subTitle = value;
    }
    return { title: copy.title, description, subTitle };
  } catch {
    return copy;
  }
}
