/** RSS catalog — grouped by region for routing world vs Israel vs site-specific queries. */

export type NewsFeedRegion =
  | "world"
  | "israel"
  | "us"
  | "europe"
  | "middle-east"
  | "asia"
  | "africa"
  | "latin-america";

export type NewsFeedDef = {
  key: string;
  url: string;
  /** Alternate URLs tried when primary fails (CORS / 404). */
  fallbackUrls?: string[];
  label: string;
  tag: string;
  region: NewsFeedRegion;
  /** Lower = higher priority within region */
  priority: number;
};

export const NEWS_FEEDS: Record<string, NewsFeedDef> = {
  // —— World / international (English) ——
  bbc: {
    key: "bbc",
    url: "https://feeds.bbci.co.uk/news/rss.xml",
    label: "BBC News",
    tag: "BBC",
    region: "world",
    priority: 1,
  },
  cnn: {
    key: "cnn",
    url: "http://rss.cnn.com/rss/edition.rss",
    label: "CNN",
    tag: "CNN",
    region: "world",
    priority: 2,
  },
  reuters: {
    key: "reuters",
    url: "https://feeds.reuters.com/reuters/worldNews",
    fallbackUrls: ["https://feeds.skynews.com/feeds/rss/world.xml", "https://www.theguardian.com/world/rss"],
    label: "Reuters World",
    tag: "Reuters",
    region: "world",
    priority: 3,
  },
  guardian: {
    key: "guardian",
    url: "https://www.theguardian.com/world/rss",
    label: "The Guardian World",
    tag: "Guardian",
    region: "world",
    priority: 4,
  },
  ap: {
    key: "ap",
    url: "https://feeds.apnews.com/rss/topnews",
    fallbackUrls: ["https://feeds.npr.org/1001/rss.xml", "https://www.cbc.ca/cmlink/rss-topstories"],
    label: "AP News",
    tag: "AP",
    region: "us",
    priority: 5,
  },
  npr: {
    key: "npr",
    url: "https://feeds.npr.org/1001/rss.xml",
    label: "NPR",
    tag: "NPR",
    region: "us",
    priority: 6,
  },
  dw: {
    key: "dw",
    url: "https://rss.dw.com/rdf/rss-en-all",
    label: "Deutsche Welle",
    tag: "DW",
    region: "europe",
    priority: 7,
  },
  france24: {
    key: "france24",
    url: "https://www.france24.com/en/rss",
    label: "France 24",
    tag: "France24",
    region: "europe",
    priority: 8,
  },
  aljazeera: {
    key: "aljazeera",
    url: "https://www.aljazeera.com/xml/rss/all.xml",
    fallbackUrls: ["https://www.france24.com/en/rss"],
    label: "Al Jazeera",
    tag: "Al Jazeera",
    region: "middle-east",
    priority: 9,
  },
  skynews: {
    key: "skynews",
    url: "https://feeds.skynews.com/feeds/rss/world.xml",
    label: "Sky News World",
    tag: "Sky",
    region: "world",
    priority: 10,
  },
  cbc: {
    key: "cbc",
    url: "https://www.cbc.ca/cmlink/rss-topstories",
    label: "CBC Canada",
    tag: "CBC",
    region: "world",
    priority: 11,
  },
  spiegel: {
    key: "spiegel",
    url: "https://www.spiegel.de/schlagzeilen/index.rss",
    label: "Der Spiegel",
    tag: "Spiegel",
    region: "europe",
    priority: 12,
  },
  lemonde: {
    key: "lemonde",
    url: "https://www.lemonde.fr/rss/une.xml",
    label: "Le Monde",
    tag: "Le Monde",
    region: "europe",
    priority: 13,
  },
  arabnews: {
    key: "arabnews",
    url: "https://www.arabnews.com/rss.xml",
    fallbackUrls: ["https://www.aljazeera.com/xml/rss/all.xml", "https://www.france24.com/en/rss"],
    label: "Arab News",
    tag: "Arab News",
    region: "middle-east",
    priority: 14,
  },
  // —— Israel (Hebrew) ——
  ynet: {
    key: "ynet",
    url: "https://www.ynet.co.il/Integration/StoryRss2.xml",
    label: "ynet",
    tag: "ynet",
    region: "israel",
    priority: 1,
  },
  haaretz: {
    key: "haaretz",
    url: "https://www.haaretz.co.il/srv/rss---feedly-homepage",
    fallbackUrls: ["https://www.jpost.com/Rss/RssFeedsHeadlines.aspx"],
    label: "הארץ",
    tag: "Haaretz",
    region: "israel",
    priority: 2,
  },
  walla: {
    key: "walla",
    url: "https://rss.walla.co.il/feed/1?type=main",
    label: "Walla",
    tag: "Walla",
    region: "israel",
    priority: 3,
  },
  mako: {
    key: "mako",
    url: "https://www.mako.co.il/rss/rss.xml",
    fallbackUrls: ["https://rss.walla.co.il/feed/1?type=main"],
    label: "N12 (mako)",
    tag: "mako",
    region: "israel",
    priority: 4,
  },
  kan: {
    key: "kan",
    url: "https://www.kan.org.il/media/fruhrsb2/rss/rss.xml",
    fallbackUrls: ["https://www.timesofisrael.com/feed/"],
    label: "כאן 11",
    tag: "Kan",
    region: "israel",
    priority: 5,
  },
  globes: {
    key: "globes",
    url: "https://www.globes.co.il/webservice/rss/rssfeeder.asmx/FeederNode?iID=585",
    label: "גלובס",
    tag: "Globes",
    region: "israel",
    priority: 6,
  },
  israelhayom: {
    key: "israelhayom",
    url: "https://www.israelhayom.co.il/rss.xml",
    label: "ישראל היום",
    tag: "Israel Hayom",
    region: "israel",
    priority: 7,
  },
  jpost: {
    key: "jpost",
    url: "https://www.jpost.com/Rss/RssFeedsHeadlines.aspx",
    label: "Jerusalem Post",
    tag: "JPost",
    region: "israel",
    priority: 8,
  },
  timesofisrael: {
    key: "timesofisrael",
    url: "https://www.timesofisrael.com/feed/",
    label: "Times of Israel",
    tag: "TOI",
    region: "israel",
    priority: 9,
  },
  themarker: {
    key: "themarker",
    url: "https://www.themarker.com/cmlink/1.144",
    label: "TheMarker",
    tag: "TheMarker",
    region: "israel",
    priority: 10,
  },
  geektime: {
    key: "geektime",
    url: "https://www.geektime.co.il/feed/",
    fallbackUrls: ["https://www.tgspot.co.il/feed/"],
    label: "Geektime",
    tag: "Geektime",
    region: "israel",
    priority: 11,
  },
  tgspot: {
    key: "tgspot",
    url: "https://www.tgspot.co.il/feed/",
    label: "TGspot",
    tag: "TGspot",
    region: "israel",
    priority: 12,
  },
  one: {
    key: "one",
    url: "https://www.one.co.il/cat/coop/xml/rss/newsfeed.aspx",
    fallbackUrls: ["https://feeds.bbci.co.uk/sport/rss.xml", "https://www.ynet.co.il/Integration/StoryRss2.xml"],
    label: "ONE",
    tag: "ONE",
    region: "israel",
    priority: 13,
  },
  sport5: {
    key: "sport5",
    url: "https://www.sport5.co.il/rss.aspx?FolderID=64",
    fallbackUrls: ["https://feeds.bbci.co.uk/sport/rss.xml", "https://www.ynet.co.il/Integration/StoryRss2.xml"],
    label: "Sport5",
    tag: "Sport5",
    region: "israel",
    priority: 14,
  },
  bloomberg: {
    key: "bloomberg",
    url: "https://feeds.bloomberg.com/markets/news.rss",
    label: "Bloomberg Markets",
    tag: "Bloomberg",
    region: "world",
    priority: 15,
  },
  ft: {
    key: "ft",
    url: "https://www.ft.com/?format=rss",
    label: "Financial Times",
    tag: "FT",
    region: "world",
    priority: 16,
  },
  cnbc: {
    key: "cnbc",
    url: "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    label: "CNBC",
    tag: "CNBC",
    region: "world",
    priority: 17,
  },
  techcrunch: {
    key: "techcrunch",
    url: "https://techcrunch.com/feed/",
    label: "TechCrunch",
    tag: "TechCrunch",
    region: "world",
    priority: 18,
  },
  theverge: {
    key: "theverge",
    url: "https://www.theverge.com/rss/index.xml",
    label: "The Verge",
    tag: "The Verge",
    region: "world",
    priority: 19,
  },
  arstechnica: {
    key: "arstechnica",
    url: "https://feeds.arstechnica.com/arstechnica/index",
    label: "Ars Technica",
    tag: "Ars",
    region: "world",
    priority: 20,
  },
  wired: {
    key: "wired",
    url: "https://www.wired.com/feed/rss",
    label: "Wired",
    tag: "Wired",
    region: "world",
    priority: 21,
  },
  mittr: {
    key: "mittr",
    url: "https://www.technologyreview.com/feed/",
    label: "MIT Technology Review",
    tag: "MIT Tech Review",
    region: "world",
    priority: 22,
  },
  openai: {
    key: "openai",
    url: "https://openai.com/news/rss.xml",
    label: "OpenAI News",
    tag: "OpenAI",
    region: "world",
    priority: 23,
  },
  huggingfaceblog: {
    key: "huggingfaceblog",
    url: "https://huggingface.co/blog/feed.xml",
    label: "Hugging Face Blog",
    tag: "Hugging Face",
    region: "world",
    priority: 24,
  },
  deepmind: {
    key: "deepmind",
    url: "https://deepmind.google/blog/rss.xml",
    label: "Google DeepMind Blog",
    tag: "DeepMind",
    region: "world",
    priority: 25,
  },
  anthropic: {
    key: "anthropic",
    url: "https://www.anthropic.com/news/rss.xml",
    fallbackUrls: ["https://openai.com/news/rss.xml", "https://techcrunch.com/feed/"],
    label: "Anthropic News",
    tag: "Anthropic",
    region: "world",
    priority: 26,
  },
  nasa: {
    key: "nasa",
    url: "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    label: "NASA",
    tag: "NASA",
    region: "world",
    priority: 27,
  },
  esa: {
    key: "esa",
    url: "https://www.esa.int/rssfeed/Our_Activities/Space_News",
    label: "ESA Space News",
    tag: "ESA",
    region: "europe",
    priority: 28,
  },
  spacecom: {
    key: "spacecom",
    url: "https://www.space.com/feeds/all",
    label: "Space.com",
    tag: "Space.com",
    region: "world",
    priority: 29,
  },
  sciencedaily: {
    key: "sciencedaily",
    url: "https://www.sciencedaily.com/rss/top/science.xml",
    label: "ScienceDaily",
    tag: "ScienceDaily",
    region: "world",
    priority: 30,
  },
  ign: {
    key: "ign",
    url: "https://feeds.ign.com/ign/all",
    fallbackUrls: ["https://www.gamespot.com/feeds/mashup/"],
    label: "IGN",
    tag: "IGN",
    region: "world",
    priority: 31,
  },
  gamespot: {
    key: "gamespot",
    url: "https://www.gamespot.com/feeds/mashup/",
    label: "GameSpot",
    tag: "GameSpot",
    region: "world",
    priority: 32,
  },
  rollingstone: {
    key: "rollingstone",
    url: "https://www.rollingstone.com/feed/",
    label: "Rolling Stone",
    tag: "Rolling Stone",
    region: "world",
    priority: 33,
  },
  variety: {
    key: "variety",
    url: "https://variety.com/feed/",
    label: "Variety",
    tag: "Variety",
    region: "world",
    priority: 34,
  },
  redditpopular: {
    key: "redditpopular",
    url: "https://www.reddit.com/r/worldnews/.rss",
    fallbackUrls: ["https://www.reddit.com/r/popular/.rss"],
    label: "Reddit Popular",
    tag: "Reddit",
    region: "world",
    priority: 35,
  },
  producthunt: {
    key: "producthunt",
    url: "https://www.producthunt.com/feed",
    label: "Product Hunt",
    tag: "Product Hunt",
    region: "world",
    priority: 36,
  },
  githubtrending: {
    key: "githubtrending",
    url: "https://mshibanami.github.io/GitHubTrendingRSS/daily/all.xml",
    label: "GitHub Trending RSS",
    tag: "GitHub Trending",
    region: "world",
    priority: 37,
  },
};

/** Headline pickers — world queries must NOT start with ynet. */
export const WORLD_HEADLINE_KEYS = [
  "bbc",
  "cnn",
  "guardian",
  "skynews",
  "npr",
  "dw",
  "france24",
  "cbc",
] as const;

export const WORLD_DIGEST_KEYS = ["bbc", "cnn", "guardian", "npr", "dw", "skynews"] as const;

export const ISRAEL_NEWS_KEYS = [
  "ynet",
  "walla",
  "israelhayom",
  "jpost",
  "timesofisrael",
  "globes",
  "themarker",
  "tgspot",
] as const;

export const ISRAEL_DIGEST_KEYS = ["ynet", "walla", "israelhayom", "jpost", "timesofisrael"] as const;

export const ISRAEL_BUSINESS_KEYS = ["globes", "themarker"] as const;
export const ISRAEL_TECH_KEYS = ["geektime", "tgspot"] as const;
export const ISRAEL_SPORT_KEYS = ["one", "sport5"] as const;

export const WORLD_BUSINESS_KEYS = ["reuters", "bloomberg", "ft", "cnbc"] as const;
export const WORLD_TECH_KEYS = ["techcrunch", "theverge", "arstechnica", "wired", "mittr"] as const;
export const AI_NEWS_KEYS = ["openai", "anthropic", "deepmind", "huggingfaceblog", "techcrunch", "mittr"] as const;
export const SPACE_SCIENCE_KEYS = ["nasa", "esa", "spacecom", "sciencedaily"] as const;
export const ENTERTAINMENT_KEYS = ["ign", "gamespot", "rollingstone", "variety"] as const;
export const TREND_KEYS = ["redditpopular", "producthunt", "githubtrending"] as const;

export const getFeedByKey = (key: string): NewsFeedDef | undefined => NEWS_FEEDS[key];

export const feedKeyFromSourceLabel = (label: string): string | null => {
  const inner = label.match(/^חדשות \((.*)\)$/)?.[1]?.trim();
  if (!inner) return null;
  for (const feed of Object.values(NEWS_FEEDS)) {
    if (feed.label === inner || feed.tag === inner) return feed.key;
  }
  return null;
};

export const sortFeedKeysByPriority = (keys: string[]): string[] =>
  [...keys].sort((a, b) => {
    const fa = NEWS_FEEDS[a];
    const fb = NEWS_FEEDS[b];
    if (!fa || !fb) return 0;
    if (fa.region !== fb.region) {
      if (fa.region === "world") return -1;
      if (fb.region === "world") return 1;
    }
    return fa.priority - fb.priority;
  });
