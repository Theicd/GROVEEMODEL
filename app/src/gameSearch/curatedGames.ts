import type { GameCategoryId, OnlineGame } from "./types";



/** Editor picks — merged into results and boosted in ranking. */

export const CURATED_GAMES: OnlineGame[] = [

  {

    id: "archive-arcade_umk3",

    title: "Ultimate Mortal Kombat 3",

    description: "Arcade fighting classic — browser playable.",

    thumbnail: "https://archive.org/services/img/arcade_umk3",

    playUrl: "https://archive.org/details/arcade_umk3",

    embedUrl: "https://archive.org/embed/arcade_umk3",

    source: "archive",

    gameType: "online",

    genre: "Fighting",

    platform: "Arcade",

    year: 1994,

    downloads: 54000,

    rating: 4.7,

    reviewsCount: 8,

    curated: true,

  },

  {

    id: "archive-msdos_Wolfenstein_3D_1992",

    title: "Wolfenstein 3D",

    description: "הטירה הנאצית — FPS קult מ-1992.",

    thumbnail: "https://archive.org/services/img/msdos_Wolfenstein_3D_1992",

    playUrl: "https://archive.org/details/msdos_Wolfenstein_3D_1992",

    embedUrl: "https://archive.org/embed/msdos_Wolfenstein_3D_1992",

    source: "archive",

    gameType: "online",

    genre: "FPS",

    platform: "PC/DOS",

    year: 1992,

    downloads: 1400000,

    curated: true,

  },

  {

    id: "archive-msdos_Maniac_Mansion_1987",

    title: "Maniac Mansion",

    description: "אחוזת המטורפים — הרפתקת LucasArts מ-1987.",

    thumbnail: "https://archive.org/services/img/msdos_Maniac_Mansion_1987",

    playUrl: "https://archive.org/details/msdos_Maniac_Mansion_1987",

    embedUrl: "https://archive.org/embed/msdos_Maniac_Mansion_1987",

    source: "archive",

    gameType: "online",

    genre: "Adventure",

    platform: "PC/DOS",

    year: 1987,

    curated: true,

  },

  {

    id: "archive-arcade_20pacgal",

    title: "Ms. Pac-Man / Galaga",

    description: "Arcade classics combo — play in browser.",

    thumbnail: "https://archive.org/services/img/arcade_20pacgal",

    playUrl: "https://archive.org/details/arcade_20pacgal",

    embedUrl: "https://archive.org/embed/arcade_20pacgal",

    source: "archive",

    gameType: "online",

    genre: "Arcade",

    platform: "Arcade",

    year: 1981,

    downloads: 35000,

    curated: true,

  },

  {

    id: "archive-arcade_galaga",

    title: "Galaga (Arcade)",

    description: "Classic shoot-em-up arcade.",

    thumbnail: "https://archive.org/services/img/arcade_galaga",

    playUrl: "https://archive.org/details/arcade_galaga",

    embedUrl: "https://archive.org/embed/arcade_galaga",

    source: "archive",

    gameType: "online",

    genre: "Arcade",

    platform: "Arcade",

    year: 1981,

    curated: true,

  },

  {

    id: "archive-arcade_donkeykong",

    title: "Donkey Kong (Arcade)",

    description: "Nintendo arcade classic.",

    thumbnail: "https://archive.org/services/img/arcade_donkeykong",

    playUrl: "https://archive.org/details/arcade_donkeykong",

    embedUrl: "https://archive.org/embed/arcade_donkeykong",

    source: "archive",

    gameType: "online",

    genre: "Arcade",

    platform: "Arcade",

    year: 1981,

    curated: true,

  },

  {

    id: "archive-msdos_Prince_of_Persia_1990",

    title: "Prince of Persia",

    description: "Platform classic — rotoscoped animation.",

    thumbnail: "https://archive.org/services/img/msdos_Prince_of_Persia_1990",

    playUrl: "https://archive.org/details/msdos_Prince_of_Persia_1990",

    embedUrl: "https://archive.org/embed/msdos_Prince_of_Persia_1990",

    source: "archive",

    gameType: "online",

    genre: "Platform",

    platform: "PC/DOS",

    year: 1990,

    curated: true,

  },

  {

    id: "archive-arcade_sfa2",

    title: "Street Fighter Alpha 2",

    description: "Capcom fighting classic in the arcade.",

    thumbnail: "https://archive.org/services/img/arcade_sfa2",

    playUrl: "https://archive.org/details/arcade_sfa2",

    embedUrl: "https://archive.org/embed/arcade_sfa2",

    source: "archive",

    gameType: "online",

    genre: "Fighting",

    platform: "Arcade",

    year: 1996,

    curated: true,

  },

];



const CURATED_BY_CATEGORY: Partial<Record<GameCategoryId, OnlineGame[]>> = {

  fighting: CURATED_GAMES.filter((g) => g.genre === "Fighting"),

  arcade: CURATED_GAMES.filter((g) => g.platform === "Arcade"),

  dos: CURATED_GAMES.filter((g) => g.platform === "PC/DOS"),

  retro: CURATED_GAMES.filter((g) => (g.year ?? 0) <= 1995),

  featured: CURATED_GAMES,

  ps1: [],

  ps2: [],

  sony: [],

};



export const curatedForCategory = (category: GameCategoryId | null): OnlineGame[] => {

  if (!category) return CURATED_GAMES.slice(0, 4);

  return CURATED_BY_CATEGORY[category] ?? CURATED_GAMES.slice(0, 4);

};



export const curatedMatchingQuery = (query: string, limit = 4): OnlineGame[] => {

  const q = query.trim().toLowerCase();

  if (!q) return [];

  return CURATED_GAMES.filter(

    (g) =>

      g.title.toLowerCase().includes(q) ||

      g.description.toLowerCase().includes(q) ||

      q.split(/\s+/).every((w) => g.title.toLowerCase().includes(w) || g.description.toLowerCase().includes(w)),

  ).slice(0, limit);

};



export const mergeCurated = (games: OnlineGame[], extras: OnlineGame[], limit: number): OnlineGame[] => {

  const seen = new Set<string>();

  const out: OnlineGame[] = [];

  for (const g of [...extras, ...games]) {

    const key = g.id.toLowerCase();

    if (seen.has(key)) continue;

    seen.add(key);

    out.push(g);

    if (out.length >= limit) break;

  }

  return out;

};

