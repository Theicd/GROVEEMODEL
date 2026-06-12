/** Browser-playable game from Internet Archive (online only). */

export type OnlineGame = {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  playUrl: string;
  embedUrl: string;
  source: "archive";
  gameType: "online";
  genre: string;
  platform: string;
  year?: number | null;
  downloads?: number;
  rating?: number | null;
  reviewsCount?: number;
  curated?: boolean;
};

export type GameCategoryId =
  | "featured"
  | "arcade"
  | "shooter"
  | "action"
  | "rpg"
  | "strategy"
  | "racing"
  | "fighting"
  | "puzzle"
  | "sports"
  | "retro"
  | "dos"
  | "console"
  | "ps1"
  | "ps2"
  | "sony";

export type GameSearchResult = {
  games: OnlineGame[];
  query: string;
  category: GameCategoryId | null;
  latencyMs: number;
  /** False when a specific title was requested but nothing relevant was found. */
  matchFound: boolean;
};

export type ResolvedGameSearch = {
  query: string;
  year: number | null;
  yearFrom: number | null;
  yearTo: number | null;
  category: GameCategoryId | null;
  /** Browse by era/category/recommendations — no specific title. */
  browseMode: boolean;
  panelTitle: string;
};
