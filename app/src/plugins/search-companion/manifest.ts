import type { GroveePluginManifest } from "../types";

export const SEARCH_COMPANION_PLUGIN_ID = "search-companion";

export const OPENSERP_UPSTREAM_VERSION = "0.8.3";

export const SEARCH_COMPANION_MANIFEST: GroveePluginManifest = {
  id: SEARCH_COMPANION_PLUGIN_ID,
  nameHe: "GROVEE Desktop",
  shortNameHe: "ממשק מקומי + חיפוש",
  descriptionHe:
    "מתקין את GROVEE על המחשב — ממשק בדפדפן על פורט 5180, מנוע חיפוש OpenSERP על 7000. בלי Node.js: הורדה, התקנה, ולחיצה על אייקון «GROVEE» בשולחן העבודה.",
  icon: "🔎",
  category: "search",
  version: "1.1.0",
  defaultBaseUrl: "http://127.0.0.1:7000",
  defaultPort: 7000,
  installStepsHe: [
    "לחץ «הורדה ל-Windows» ושמור את GroveDesktop-Setup.exe.",
    "הרץ את קובץ ההתקנה — בחר תיקייה (או השאר ברירת מחדל).",
    "בסיום — לחץ על אייקון «GROVEE» בשולחן העבודה.",
    "הממשק נפתח בדפדפן — חיפוש ושיחה מקומית (מודלים יורדים בפעם הראשונה).",
  ],
  download: {
    win: {
      url: "./plugins/GroveDesktop-Setup-1.0.0.exe",
      filename: "GroveDesktop-Setup-1.0.0.exe",
      sizeHintHe: "~120 MB (+ OpenSERP ~25 MB בהתקנה)",
      fallbackUrl: "./plugins/grove-desktop-win.zip",
      fallbackFilename: "grove-desktop-win.zip",
    },
  },
  upstream: {
    name: "OpenSERP",
    version: OPENSERP_UPSTREAM_VERSION,
    url: "https://github.com/karust/openserp",
  },
};
