# GROVEE — Game Search Checklist (Browser / GitHub Pages)

> **אין backend** — הכל `fetch` ל-Internet Archive מהדפדפן, כמו `webSearch/`.

## Phase 1 — תשתית

- [x] **1.1** `gameSearch/types.ts` — OnlineGame + year/rating/downloads
- [x] **1.2** `gameSearch/archiveQueries.ts` — PS1/PS2/Sony/DOS/console + year filter
- [x] **1.3** `gameSearch/archiveBrowser.ts` — rotated pool (2×80), quality score
- [x] **1.4** `gameSearch/gameIntents.ts` + `gameAliases.ts` — PS1/PS2/Hebrew aliases
- [x] **1.5** `curatedGames.ts` + `public/games/featured.json` — MK, Wolfenstein, MM 1987
- [x] **1.6** `rotation.ts` — session page rotation per refresh
- [x] **1.7** Unit + live tests (`npm run test:games` — 6 tests)

## Phase 2 — UI

- [x] **2.1** `GameCard.tsx` — platform, year, ★ rating, downloads, TOP badge
- [x] **2.2** `GamesPanel.tsx` — 🔄 refresh + 20 games per category
- [x] **2.3** `GameSpotlight.tsx` — pool refresh every 5 min + rotation
- [x] **2.4** `GameResultsStrip.tsx` — inline chat cards
- [x] **2.5** CSS — meta, curated, spotlight sub

## Phase 3 — שילוב App

- [x] **3.1** פאנל ימין — `gamesPanelOpen`
- [x] **3.2** Intent: bored + PS1/PS2/שם משחק + שנה
- [x] **3.3** `gameResults` על הודעת assistant
- [x] **3.4** Grounding prompt
- [x] **3.5** mutual exclusive panels

## Phase 4 — Acceptance

| # | פעולה | צפוי |
|---|--------|------|
| A1 | "משחק ps1" / "סוני" | קטגוריה PS1 + משחקי psx |
| A2 | "הטירה הנאית" / "אחוזת המטורפים 1987" | Wolfenstein / Maniac Mansion |
| A3 | "mortal kombat" | UMK3 / fighting |
| A4 | 🔄 רענן / פתיחה מחדש | רשימה שונה (rotation) |
| A5 | `npm run test:games` | 6/6 Archive ok |

## Phase 5 — עתיד (לא ב-scope)

- [ ] build-time index JSON ב-CI
- [ ] lazy-load GamesPanel chunk
