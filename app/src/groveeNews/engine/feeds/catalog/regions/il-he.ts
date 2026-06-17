// @ts-nocheck
import { isBlockedFeedKey } from "../../blockedFeeds";
import type { CatalogFeed } from "../types";

function ilFeed(
  def: Omit<CatalogFeed, "lang" | "region" | "scope" | "tier" | "language" | "sourceType" | "category"> &
    Partial<Pick<CatalogFeed, "category" | "ideology" | "fallbackUrls">>,
): CatalogFeed {
  return {
    language: "multi",
    sourceType: "rss",
    category: "israel",
    lang: "he",
    region: "IL",
    scope: "national",
    tier: "core",
    ideology: "conservative",
    ...def,
  };
}

/** Israeli Hebrew sources — right / center-right editorial line. */
export const IL_HE_FEEDS: CatalogFeed[] = [
  ilFeed({
    key: "il_makor_rishon",
    url: "https://news.google.com/rss/search?q=site:makorrishon.co.il&hl=he&gl=IL&ceid=IL:he",
    fallbackUrls: ["https://www.makorrishon.co.il/feed/"],
    label: "מקור ראשון",
    tag: "מקור ראשון",
    priority: 1,
  }),
  ilFeed({
    key: "il_israel_hayom",
    url: "https://www.israelhayom.co.il/rss",
    label: "ישראל היום",
    tag: "ישראל היום",
    priority: 2,
  }),
  ilFeed({
    key: "il_now14",
    url: "https://www.now14.co.il/feed/",
    fallbackUrls: ["https://www.c14.co.il/feed/"],
    label: "ערוץ 14",
    tag: "14",
    priority: 3,
  }),
  ilFeed({
    key: "il_inn_he",
    url: "https://www.inn.co.il/Rss.aspx",
    fallbackUrls: ["https://www.israelnationalnews.com/Rss.aspx"],
    label: "ערוץ 7",
    tag: "INN",
    priority: 4,
  }),
  ilFeed({
    key: "il_srugim",
    url: "https://www.srugim.co.il/feed/",
    label: "סרוגים",
    tag: "סרוגים",
    priority: 5,
  }),
  ilFeed({
    key: "il_besheva",
    url: "https://www.besheva.co.il/feed/",
    fallbackUrls: ["https://www.besheva.co.il/rss"],
    label: "בשבע",
    tag: "בשבע",
    priority: 6,
  }),
  ilFeed({
    key: "il_israel_defense",
    url: "https://news.google.com/rss/search?q=site:israeldefense.co.il&hl=he&gl=IL&ceid=IL:he",
    label: "Israel Defense",
    tag: "ID",
    priority: 7,
    category: "world",
  }),
  ilFeed({
    key: "il_mida",
    url: "https://www.mida.org.il/feed/",
    label: "מידה",
    tag: "מידה",
    priority: 8,
    ideology: "center-right",
  }),
  ilFeed({
    key: "il_hakol_yehudi",
    url: "https://news.google.com/rss/search?q=site:hakolhayehudi.co.il&hl=he&gl=IL&ceid=IL:he",
    label: "הקול היהודי",
    tag: "הקול היהודי",
    priority: 9,
  }),
  ilFeed({
    key: "il_globes_he",
    url: "https://www.globes.co.il/webservice/rss/rssfeeder.asmx/FeederNode?iID=2",
    fallbackUrls: [
      "https://www.globes.co.il/webservice/rss/rssfeeder.asmx/FeederNode?iID=585",
    ],
    label: "גלובס",
    tag: "גלובס",
    priority: 10,
    category: "business",
    ideology: "center-right",
  }),
  ilFeed({
    key: "il_0404",
    url: "https://news.google.com/rss/search?q=site:0404.co.il&hl=he&gl=IL&ceid=IL:he",
    label: "0404",
    tag: "0404",
    priority: 11,
  }),
].filter((f) => !isBlockedFeedKey(f.key));
