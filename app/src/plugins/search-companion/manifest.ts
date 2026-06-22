import type { GroveePluginManifest } from "../types";

export const SEARCH_COMPANION_PLUGIN_ID = "search-companion";

export const OPENSERP_UPSTREAM_VERSION = "0.8.3";

export const SEARCH_COMPANION_MANIFEST: GroveePluginManifest = {
  id: SEARCH_COMPANION_PLUGIN_ID,
  nameHe: "Grove Search Companion",
  shortNameHe: "מנוע חיפוש מקומי",
  descriptionHe:
    "שירות קטן על המחשב שלך (OpenSERP) — Google/Bing/DuckDuckGo/Yandex/Ecosia, תמונות מ-Google/Bing, וקישורי וידאו (Archive/Vimeo/PeerTube). GROVEEMODEL שולח אליו בקשות ו-Gemma מסכמת בדפדפן.",
  icon: "🔎",
  category: "search",
  version: "1.0.0",
  defaultBaseUrl: "http://127.0.0.1:7000",
  defaultPort: 7000,
  installStepsHe: [
    "לחץ «הורדה ל-Windows» ושמור את הקובץ.",
    "חלץ את ה-ZIP לתיקייה (למשל Downloads\\GroveSearch).",
    "לחץ פעמיים על Install-GroveSearchCompanion.ps1 (או Run-Install.bat).",
    "הפעל «Grove Search» משולחן העבודה — האייקון יהפוך לירוק כאן.",
  ],
  download: {
    win: {
      url: "./plugins/grove-search-companion-win.zip",
      filename: "GroveSearchCompanion-win.zip",
      sizeHintHe: "~2 MB (+ OpenSERP ~25 MB בהתקנה)",
    },
  },
  upstream: {
    name: "OpenSERP",
    version: OPENSERP_UPSTREAM_VERSION,
    url: "https://github.com/karust/openserp",
  },
};
