import type { OnlineGame } from "./types";



export function archiveIdentifier(game: OnlineGame): string {

  return game.id.replace(/^archive-/, "");

}



/** Internet Archive item image (box art / screenshot when available). */

export function gameThumbnailUrl(game: OnlineGame): string {

  if (game.thumbnail?.startsWith("http")) return game.thumbnail;

  return `https://archive.org/services/img/${archiveIdentifier(game)}`;

}



/** Larger Archive tile when no explicit thumbnail — better for hero/backdrop. */

export function gameHeroImageUrl(game: OnlineGame): string {

  if (game.thumbnail?.startsWith("http")) return game.thumbnail;

  const id = archiveIdentifier(game);

  return `https://archive.org/services/img/${id}`;

}



/** Hero rotation — favorites only (git + local IndexedDB). */

export function buildHeroLineup(favorites: OnlineGame[]): OnlineGame[] {

  return favorites;

}

