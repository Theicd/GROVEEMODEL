/**

 * Live game search — Internet Archive from browser (GitHub Pages compatible).

 * Run: npm run test:games

 */

import { describe, it, expect } from "vitest";

import { searchOnlineGames, randomOnlineGames, searchFromResolved } from "./archiveBrowser";
import { resolveGameSearch } from "./gameAliases";
import { parseGameUserRequest } from "./gameIntents";



const TIMEOUT = 25_000;



describe.sequential("game search live", () => {

  it(

    "searchOnlineGames returns playable online games for pacman",

    async () => {

      const result = await searchOnlineGames("pacman", 6, "arcade");

      expect(result.games.length).toBeGreaterThan(0);

      expect(result.games[0].embedUrl).toContain("archive.org/embed/");

      expect(result.games[0].gameType).toBe("online");

    },

    TIMEOUT,

  );



  it(

    "randomOnlineGames returns large rotated pool",

    async () => {

      const a = await randomOnlineGames(12, "featured");

      const b = await randomOnlineGames(12, "featured");

      expect(a.games.length).toBeGreaterThan(0);

      const aIds = new Set(a.games.map((g) => g.id));

      const overlap = b.games.filter((g) => aIds.has(g.id)).length;

      expect(overlap).toBeLessThan(b.games.length);

    },

    TIMEOUT,

  );



  it(

    "PS1 category returns PlayStation browser games",

    async () => {

      const result = await searchOnlineGames("", 6, "ps1");

      expect(result.games.length).toBeGreaterThan(0);

      expect(result.games.some((g) => g.platform.includes("PlayStation"))).toBe(true);

    },

    TIMEOUT,

  );



  it(

    "Hebrew alias resolves Wolfenstein and finds games",

    async () => {

      const resolved = resolveGameSearch("הטירה הנאית", null);

      expect(resolved.query).toContain("wolfenstein");

      const result = await searchOnlineGames(resolved.query, 4, resolved.category);

      expect(result.games.length).toBeGreaterThan(0);

    },

    TIMEOUT,

  );



  it(

    "Maniac Mansion 1987 search finds adventure game",

    async () => {

      const resolved = resolveGameSearch("אחוזת המטורפים 1987", null);

      const result = await searchOnlineGames(resolved.query, 4, "dos", resolved.year);

      expect(result.games.length).toBeGreaterThan(0);

    },

    TIMEOUT,

  );



  it(

    "Mortal Kombat fighting search returns results",

    async () => {

      const result = await searchOnlineGames("mortal kombat", 6, "fighting");

      expect(result.games.length).toBeGreaterThan(0);

      expect(result.games.some((g) => /mortal|kombat|mk/i.test(g.title))).toBe(true);

    },

    TIMEOUT,

  );

  it(
    "80s arcade browse returns games",
    async () => {
      const req = parseGameUserRequest("חפש משחקים משנות ה80 בקטגוריית ארקייד");
      const result = await searchFromResolved(req, 8);
      expect(result.games.length).toBeGreaterThan(0);
    },
    TIMEOUT,
  );

});

