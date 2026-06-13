# GROVEE Web Search — Implementation Plan & Checklist

> מטרה: חיפוש חכם, מהיר ואמין — עם מקורות חינמיים שנבדקו לפני חיבור.

---

## Phase 0 — אבחון (הבעיה המקורית)

- [x] **0.1** זיהוי: `fetchWebContext` הישן חיפש רק Wikipedia + GitHub (מוגבל)
- [x] **0.2** שאלות בעברית ("מזג אוויר בניו יורק") לא הפעילו GitHub ולא קיבלו מזג אוויר
- [x] **0.3** Wikipedia snippets קצרים מדי לשאלות factual
- [x] **0.4** אין הצגת מקורות למשתמש — רק הקשר נסתר למודל

---

## Phase 1 — תשתית (הושלם)

- [x] **1.1** מודול `webSearch/` — types, intents, fetchJson עם timeout
- [x] **1.2** Router לפי כוונה: weather, marine, earthquake, github, huggingface, wikipedia
- [x] **1.3** Orchestrator מקביל (`Promise.all`) — זמן יעד &lt; 5 שניות
- [x] **1.4** בדיקות יחידה (intents + format + mocks)
- [x] **1.5** חיבור ל-`App.tsx` + UI מקורות
- [x] **1.6** `userRequestsSearch` — מילות "חפש/חיפוש" מפעילות גם בלי toggle

---

## Phase 2 — מקורות מחוברים (Phase 1)

| # | מקור | API | סטטוס | בדיקה |
|---|------|-----|--------|--------|
| 2.1 | **Open-Meteo** (מזג אוויר) | `geocoding-api.open-meteo.com` + `api.open-meteo.com` | ✅ מחובר | שאלה: "מה מזג האוויר בניו יורק" |
| 2.2 | **Open-Meteo Marine** (גלים) | `marine-api.open-meteo.com` | ✅ מחובר | שאלה: "גובה גלים בתל אביב" |
| 2.3 | **USGS** (רעידות אדמה) | `earthquake.usgs.gov/.../all_day.geojson` | ✅ מחובר | שאלה: "רעידות אדמה אחרונות" |
| 2.4 | **Wikipedia EN+HE** | MediaWiki API + extract | ✅ משודרג | שאלה כללית / היסטוריה |
| 2.5 | **GitHub Repos** | `api.github.com/search/repositories` | ✅ משודרג | שאילתות טech / open source |
| 2.6 | **Hugging Face** | `huggingface.co/api/models` + datasets | ✅ מחובר | "מודלים לעברית" |

---

## Phase 3 — מקורות מורחבים (Phase 2.5 — הושלם)

| # | מקור | API | סטטוס | דוגמת שאלה |
|---|------|-----|--------|------------|
| 3.0 | **World Time API** | `worldtimeapi.org` + Open-Meteo geocoding | ✅ | "מה השעה בטוקיו" |
| 3.1 | **REST Countries** | `restcountries.com/v3.1` | ✅ | "מה הבירה של גרמניה" |
| 3.2 | **Nager.Date** | `date.nager.at/api/v3` | ✅ | "האם היום חג בגרמניה" |
| 3.3 | **Wikidata SPARQL** | `query.wikidata.org` | ✅ | "מי ראש הממשלה של ישראל" |
| 3.4 | **Frankfurter FX** | `api.frankfurter.app` | ✅ | "USD to ILS" |

## Phase 4 — מקורות עתידיים (TODO)

- [ ] **4.1** Open-Meteo Air Quality
- [ ] **4.2** Hacker News / arXiv
- [ ] **4.3** CoinGecko (קריפטו) — rate limit
- [ ] **4.4** Finnhub / Alpha Vantage (בורסות) — דורש API key
- [ ] **4.5** OpenSky (טיסות) — CORS + rate limits
- [ ] **4.6** NASA APOD / NeoWs (חלל)
- [ ] **4.7** GDELT / RSS חדשות — proxy CORS
- [ ] **4.8** Data.gov.il / data.gov — ממשל פתוח לפי מדינה
- [ ] **4.9** **Knowledge Graph פנימי** — cache מדינה→{PM, בירה, מטבע, חגים, מז"א}
- [ ] **4.10** Cache 5 דקות + fast-path תשובה ישירה למז"א/שעון

> **כלל:** לפני חיבור — `curl`/test + vitest mock + בדיקה ידנית מהדפדפן.

---

## Phase 4 — UX & איכות תשובה

- [x] **4.1** הצגת בלוק "מקורות חיפוש" בצ'אט (לפני תשובת המודל)
- [x] **4.2** סטטוס: "Searching…" + סיכום מקורות בהצלחה/כישלון
- [x] **4.3** Grounding prompt — המודל חייב להשתמש בנתוני החיפוש
- [ ] **4.4** תשובה ישירה למזג אוויר **בלי המתנה למודל** (fast path) — עתידי
- [ ] **4.5** Cache 5 דקות לשאילתות זהות — עתידי

---

## Phase 5 — בדיקות acceptance

| בדיקה | צפוי |
|--------|------|
| Search ON + "מה מזג האוויר בניו יורק?" | טמפרatura, רוח, תחזית + מקור Open-Meteo |
| Search ON + "רעידות אדמה ביפן" | רשימת USGS מסוננת |
| Search ON + "github llm chat" | מאגרי GitHub + כוכבים |
| Search ON + "huggingface gemma" | מודלים מ-HF |
| "חפש מידע על פירמידות" (בלי toggle) | Wikipedia |
| קובץ מצורף + Search ON | חיפוש **לא** רץ (כמו קודם) |
| אין אינטרנט | הודעת כישלון ברורה |

---

## ארכיטקטורה

```
שאלת משתמש + Search toggle / "חפש"
        ↓
 classifySearchIntents()
        ↓
 ┌──────┴──────┬──────────┬─────────┬──────────┬─────────┐
 │ Open-Meteo  │ USGS     │ Wiki    │ WorldTime│ Countries│ …
 └──────┬──────┴──────────┴─────────┴──────────┴─────────┘
        ↓
 formatWebContext() → system prompt
 searchSources → UI block
        ↓
 Gemma מסכם עם grounding
```

---

## הערות CORS / Rate limits

- **Open-Meteo, USGS, Wikipedia, HF** — עובדים מהדפדפן (נבדק)
- **GitHub** — 60 req/h ללא token; token אופציונלי בהגדרות (עתידי)
- **RSS** — רוב הפידים חוסמים CORS → דורש proxy

---

*עודכן: Phase 1 implemented in `app/src/webSearch/`.*
