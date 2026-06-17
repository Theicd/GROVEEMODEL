// @ts-nocheck
import type { CatalogFeed } from "./types";

function worldFeed(
  def: Omit<CatalogFeed, "lang" | "region" | "scope" | "tier" | "language" | "sourceType" | "category"> &
    Pick<CatalogFeed, "lang"> &
    Partial<Pick<CatalogFeed, "category" | "fallbackUrls">>,
): CatalogFeed {
  return {
    language: "multi",
    sourceType: "rss",
    category: "world",
    region: "GLOBAL",
    scope: "global",
    tier: "core",
    ...def,
  };
}

/** Non-English sources polled for all users — matched by lang metadata, not per-country packs. */
export const WORLD_LANG_FEEDS: CatalogFeed[] = [
  // Hebrew (Israel)
  worldFeed({
    key: "il_ynet_he",
    lang: "he",
    url: "https://www.ynet.co.il/Integration/StoryRss2.xml",
    label: "Ynet",
    tag: "Ynet",
    priority: 2,
    category: "israel",
  }),
  worldFeed({
    key: "il_jpost_he",
    lang: "he",
    url: "https://news.google.com/rss/search?q=site:jpost.com&hl=he&gl=IL&ceid=IL:he",
    fallbackUrls: ["https://www.jpost.com/rss/rssfeedsheadlines"],
    label: "Jerusalem Post HE",
    tag: "JPost",
    priority: 3,
    category: "israel",
  }),
  // French
  worldFeed({
    key: "fr_lemonde",
    lang: "fr",
    url: "https://www.lemonde.fr/rss/une.xml",
    label: "Le Monde",
    tag: "Le Monde",
    priority: 2,
  }),
  worldFeed({
    key: "fr_france24",
    lang: "fr",
    url: "https://www.france24.com/fr/rss",
    label: "France 24",
    tag: "France 24",
    priority: 3,
  }),
  // German
  worldFeed({
    key: "de_spiegel",
    lang: "de",
    url: "https://www.spiegel.de/schlagzeilen/index.rss",
    label: "Der Spiegel",
    tag: "Spiegel",
    priority: 2,
  }),
  worldFeed({
    key: "de_tagesschau",
    lang: "de",
    url: "https://www.tagesschau.de/xml/rss2",
    label: "Tagesschau",
    tag: "Tagesschau",
    priority: 3,
  }),
  // Russian
  worldFeed({
    key: "ru_rt",
    lang: "ru",
    url: "https://russian.rt.com/rss",
    label: "RT Russian",
    tag: "RT",
    priority: 2,
  }),
  worldFeed({
    key: "ru_tass",
    lang: "ru",
    url: "https://tass.com/rss/v2.xml",
    label: "TASS",
    tag: "TASS",
    priority: 3,
  }),
  // Japanese
  worldFeed({
    key: "ja_nhk",
    lang: "ja",
    url: "https://www3.nhk.or.jp/rss/news/cat0.xml",
    label: "NHK",
    tag: "NHK",
    priority: 2,
  }),
  worldFeed({
    key: "ja_asahi",
    lang: "ja",
    url: "https://www.asahi.com/rss/asahi/newsheadlines.rdf",
    label: "Asahi Shimbun",
    tag: "Asahi",
    priority: 3,
  }),
  // Chinese
  worldFeed({
    key: "zh_scmp",
    lang: "zh",
    url: "https://news.google.com/rss/search?q=site:scmp.com&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
    fallbackUrls: ["https://www.scmp.com/rss/91/feed"],
    label: "SCMP",
    tag: "SCMP",
    priority: 2,
  }),
  worldFeed({
    key: "zh_caixin",
    lang: "zh",
    url: "https://news.google.com/rss/search?q=site:caixin.com&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
    label: "Caixin",
    tag: "Caixin",
    priority: 3,
  }),
  // Spanish
  worldFeed({
    key: "es_elpais",
    lang: "es",
    url: "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    label: "El País",
    tag: "El País",
    priority: 2,
  }),
  worldFeed({
    key: "es_elmundo",
    lang: "es",
    url: "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    label: "El Mundo",
    tag: "El Mundo",
    priority: 3,
  }),
  // Arabic
  worldFeed({
    key: "ar_arabnews",
    lang: "ar",
    url: "https://www.arabnews.com/rss.xml",
    label: "Arab News",
    tag: "Arab News",
    priority: 2,
  }),
  // Korean
  worldFeed({
    key: "ko_yonhap",
    lang: "ko",
    url: "https://www.yonhapnews.co.kr/rss/english.xml",
    fallbackUrls: ["https://en.yna.co.kr/rss/industry.xml"],
    label: "Yonhap",
    tag: "Yonhap",
    priority: 2,
  }),
  // Italian
  worldFeed({
    key: "it_ansa",
    lang: "it",
    url: "https://www.ansa.it/english/news/english_rss.xml",
    label: "ANSA",
    tag: "ANSA",
    priority: 2,
  }),
  // Portuguese (Brazil)
  worldFeed({
    key: "pt_g1",
    lang: "pt",
    url: "https://g1.globo.com/rss/g1/",
    label: "G1 Globo",
    tag: "G1",
    priority: 2,
  }),
  // Ukrainian
  worldFeed({
    key: "uk_kyiv",
    lang: "uk",
    url: "https://kyivindependent.com/feed/",
    label: "Kyiv Independent",
    tag: "Kyiv Independent",
    priority: 2,
  }),
];
