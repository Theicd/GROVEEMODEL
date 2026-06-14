# Startup Context — תוכנית יישום GROVEE

> שכבת הקשר בסיסית: זמן + timezone + מיקום (IP) — קריאה אחת בפתיחה, שימוש בכל הממשק.

## מטרות

1. **שעון/תאריך אזוריים** בממשק — בלי לשאול את המשתמש
2. **עולם חי (REALITY LIVE)** — נפתח על המיקום שלך (גרמניה → גרמניה, תל אביב → תל אביב)
3. **חיסכון בחיפושים** — "מה השעה?", "איזה יום", "מזג אוויר?" → context מקומי, לא רשת
4. **POI באזור** — "מסעדה באזור" → anchor מ-context
5. **Attribution** — קישור Time.Now ב-About

---

## Phase 0 — תשתית ✅

- [x] `startupContext/types.ts` — טיפוס `StartupContext`
- [x] `startupContext/fetchStartupContext.ts` — Time.Now `/ip` + geo (ipapi) + cache
- [x] `startupContext/localTime.ts` — תשובות זמן/תאריך בלי API
- [x] `startupContext/promptBlock.ts` — בלוק הקשר ל-system prompt
- [x] `proxyFetch.ts` — `time.now` ב-CORS direct
- [ ] בדיקת CORS מ-GitHub Pages (live)

## Phase 1 — ממשק (UI)

- [x] `LocalContextBar.tsx` — שעה חיה + תאריך + טמפרatura מקומית
- [x] CSS `.local-context-bar` ב-`index.css`
- [x] שילוב ב-`chat-header` (ימין, לפני כפתורי פעולה)
- [ ] אנימציית טמפרatura (pulse קל כשמתעדכן)
- [x] Attribution Time.Now ב-info modal

## Phase 2 — עולם חי / מפה

- [x] `GlobePanel` — `getStartupContext()` במקום geo נפרד
- [x] `flyTo` על lat/lon בפתיחה (zoom עיר)
- [x] `setUserRegion` עם country + coords מ-context
- [ ] reality iframe — וידוא שהמפה מגיבה ל-flyTo (QA ידני)

## Phase 3 — חיפוש חכם (חיסכון רשת)

- [x] `isLocalContextTimeQuery` — שאלות זמן/תאריך **ללא** עיר → **לא** `needsWebSearch`
- [x] `buildLocalTimeAnswer` — הזרקה ל-webContext בלי providers
- [x] `isNearMeAnchor` + `resolveNearAnchor` — "באזור"/"קרוב אליי"
- [x] `worldTime.ts` — fallback מיקום מ-context
- [x] `openMeteo.ts` — מזג אוויר בלי עיר → coords מ-context
- [x] `nominatimPlaces.ts` — POI + anchor מ-context
- [x] `extractTimeZonePair` — צד "ישראל"/context → context timezone
- [ ] "מה השעה בכל מדינה" — רשימה סטטית + timezone list (עתידי)

## Phase 4 — בדיקות ו-deploy

- [x] Unit tests — `startupContext.test.ts`, intents
- [ ] `acceptanceQueries` — QA-T05 "מה השעה?" ללא עיר
- [ ] `npm run build:pages-docs` + push

---

## סדר עדיפויות שאילתות (ללא חיפוש ברשת)

| שאלה | מקור | חיפוש? |
|------|------|--------|
| מה השעה? / מה השעה עכשיו? | StartupContext | ❌ |
| איזה יום / מה התאריך? | StartupContext | ❌ |
| מה השעה בטוקיו? | world-time provider | ✅ 1 |
| הפרש ישראל–לונדון | world-time (ישראל=context) | ✅ 1 geocode |
| מה מזג האוויר? | Open-Meteo + context coords | ✅ 1 |
| מסעדה באזור | Nominatim + context city | ✅ 1 |
| מחיר ביטקוין | coingecko | ✅ |

---

## Attribution (חובה Time.Now)

```html
<a href="https://time.now">World Time API by Time.Now</a>
```

מופיע ב: Info modal → "מקורות מידע חיים".
