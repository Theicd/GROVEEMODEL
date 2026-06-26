import { translateTexts } from "../../groveeNews/engine/translate/googleTranslate";
import type { ChatUiLanguage } from "../../ui/useUiLanguage";
import { needsHebrewTranslation } from "../../tmdb/tmdbLocalize";

export type LocalizableEpgCopy = {
  title: string;
  description?: string | null;
  subTitle?: string | null;
};

/** Translate EPG programme copy for Hebrew UI when text is still Latin. */
export async function localizeEpgCopyForUi(
  copy: LocalizableEpgCopy,
  uiLang: ChatUiLanguage,
): Promise<LocalizableEpgCopy> {
  if (uiLang !== "he") return copy;

  const pending: Array<"title" | "description" | "subTitle"> = [];
  const texts: string[] = [];
  if (needsHebrewTranslation(copy.title)) {
    pending.push("title");
    texts.push(copy.title);
  }
  if (needsHebrewTranslation(copy.description)) {
    pending.push("description");
    texts.push(copy.description!);
  }
  if (needsHebrewTranslation(copy.subTitle)) {
    pending.push("subTitle");
    texts.push(copy.subTitle!);
  }
  if (!texts.length) return copy;

  try {
    const { texts: translated } = await translateTexts(texts, "he", "en");
    let idx = 0;
    let title = copy.title;
    let description = copy.description;
    let subTitle = copy.subTitle;
    for (const field of pending) {
      const value = translated[idx++]?.trim();
      if (!value) continue;
      if (field === "title") title = value;
      else if (field === "description") description = value;
      else subTitle = value;
    }
    return { title, description, subTitle };
  } catch {
    return copy;
  }
}
