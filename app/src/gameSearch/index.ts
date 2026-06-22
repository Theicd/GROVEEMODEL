export type { GameCategoryId, GameSearchResult, OnlineGame, ResolvedGameSearch } from "./types";
export { GAME_CATEGORIES, FEATURED_ROTATION_POOL } from "./archiveQueries";
export {
  resolveGameSearch,
  formatPopularityLabel,
  extractDecadeRange,
  buildGamePanelTitle,
} from "./gameAliases";
export { CURATED_GAMES } from "./curatedGames";
export {
  searchOnlineGames,
  searchFromResolved,
  randomOnlineGames,
  searchOnlineGamesWithFallback,
  loadFeaturedFallback,
  archiveIdentifierFromGame,
} from "./archiveBrowser";
export {
  isGameSearchRequest,
  extractGameQuery,
  parseGameUserRequest,
  detectGameCategory,
  shouldOpenGamePanel,
  buildGameSearchPanelTitle,
  detectCategoryFromText,
  categoryLabelHe,
  formatCategoryListForPrompt,
  extractUserIntentPrefix,
  isTextCompositionRequest,
  isInlineTextTaskRequest,
  isTextTransformRequest,
  getIntentScanText,
} from "./gameIntents";
export { buildGameSearchFoundReply, buildGameSearchNotFoundReply } from "./gameReplyMessages";
