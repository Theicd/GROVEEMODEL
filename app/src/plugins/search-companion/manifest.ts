import type { GroveePluginManifest } from "../types";

export const SEARCH_COMPANION_PLUGIN_ID = "search-companion";

export const OPENSERP_UPSTREAM_VERSION = "0.8.3";

export const SEARCH_COMPANION_MANIFEST: GroveePluginManifest = {
  id: SEARCH_COMPANION_PLUGIN_ID,
  nameHe: "Grove Search Companion",
  shortNameHe: "חיפוש מקומי",
  descriptionHe:
    "מנוע חיפוש OpenSERP על המחשב (פורט 7000). מריצים Start-GroveSearch.bat מתוך GroVeeSerch או מהקיצור בשולחן העבודה.",
  icon: "🔎",
  category: "search",
  version: "1.1.0",
  defaultBaseUrl: "http://127.0.0.1:7000",
  defaultPort: 7000,
  installStepsHe: [
    "הורד חבילת ZIP (או הרץ Run-Install.bat מתוך GroVeeSerch).",
    "הרץ Start-GroveSearch.bat - מנוע החיפוש עולה על פורט 7000.",
    "פתח GROVEEMODEL ולחץ תוספים - לחץ בדוק חיבור.",
    "כשהסטטוס ירוק, חיפוש web מקומי זמין בשיחה.",
  ],
  download: {
    win: {
      url: "./plugins/grove-search-companion-win.zip",
      filename: "grove-search-companion-win.zip",
      sizeHintHe: "~25 MB (OpenSERP + סקריפטי הפעלה)",
      fallbackUrl: "./plugins/GroveDesktop-Setup-1.0.0.exe",
      fallbackFilename: "GroveDesktop-Setup-1.0.0.exe",
    },
  },
  upstream: {
    name: "OpenSERP",
    version: OPENSERP_UPSTREAM_VERSION,
    url: "https://github.com/karust/openserp",
  },
};
