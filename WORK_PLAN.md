# GROVEEMODEL — תוכנית עבודה מרכזית

**פרויקט:** `c:\Users\Avatar001\CascadeProjects\GROVEEMODEL`  
**מקור:** סיכום השיחה המלאה (ארכיטקטורה → באגים → קריסה → "אוויר בשיחה" → חיפוש חכם)  
**נוצר:** 2026-06-13  
**מצב:** יושם חלקית 2026-06-13 — **תוכנית המשך:** [`IMPROVEMENT_WORKPLAN.md`](./IMPROVEMENT_WORKPLAN.md) (Context Ring, clouad patterns, QA מלא)

---

## 0. רקע — מה ביקשת מתחילת השיחה

| # | הבקשה המקורית | סטטוס ניתוח |
|---|----------------|-------------|
| 0.1 | להבין טעינת מודל בדפדפן, הורדה, תקשורת, שמירה, הפעלה חוזרת | ✅ נותח — Web Worker + HF cache + localStorage |
| 0.2 | חיפוש ברשת יעיל + יציבות בשיחה | ⚠️ זוהו באגים מבניים |
| 0.3 | תקיעות בחיפוש / טמפרatura / מידע | ⚠️ Globe + mutex + prompt overflow |
| 0.4 | ראייה — רק כשמצלמה פעילה, לא לתקוע צ'אט | ⚠️ vision לא נעצר בפועל בזמן chat |

---

## 1. טבלת תלונות → מסקנות → שינוי נדרש

### 1.1 חלון "עולם חי" (Globe) נפתח בכל שאלת מידע

| | |
|---|---|
| **תלונה** | בכל שאלה על טמפרatura, מזג אוויר, מידע כללי — חלון המפה נפתח אוטומטית |
| **קובץ** | `App.tsx` ~2560, `realityGlobe/intents.ts` |
| **שורש הבעיה** | `shouldOpenGlobePanel()` מחזיר `true` אם **intent** של חיפוש הוא `weather`, `places`, `distance` וכו' — גם בלי בקשת מפה. Regex רחב (`איפה`, `where is`, `weather`) מחזק את זה |
| **מסקנה** | Globe צריך להיפתח רק ב**בקשת מפה מפורשת** (`isGlobePresentationQuery`) או intent גיאוגרפי + מילות הצגה |
| **שינוי** | לשנות `shouldOpenGlobePanel`: intent בלבד ≠ פתיחה. weather/time → חיפוש בלבד, Globe רק אם "הצג על המפה" / "show on map" |

---

### 1.2 שעון עולמי נכשל — "בוקר טוב… מה השעה בישראל"

| | |
|---|---|
| **תלונה** | TimeAPI נכשל; המודל אומר שאין לו נתוני זמן חיים |
| **UI** | `לא נמצא אזור זמן: מר גrובi איזhe יום...` — **כל המשפט** נשלח ל-geocoder |
| **קובץ** | `worldTime.ts`, `queryExtract.ts`, `intents.ts` |
| **שורש הבעיה** | (1) `extractLocationPhrase` נכשל על `בישrael` (regex דורש רווח אחרי `ב`). (2) fallback = `query.trim()` — כל השאלה. (3) ברכה "בוקר טוב מר גrובi" מזהמת חילוץ. (4) `worldtime` הוא structured intent → **חוסם** Wikipedia fallback |
| **מסקנה** | צריך sanitization לפני חילוץ + fallback חכם (Israel/timezone ידוע) + Wikipedia רק אם structured נכשל |
| **שינוי** | `sanitizeSearchQuery()` · תיקון regex עברית · `extractLocationPhrase` עם `בישrael`/`ישrael` · geocode fallback ל-"Israel" · אם worldtime נכשל → fast-path timezone או wiki |

---

### 1.3 רוב מקורות החיפוש נכשלים — רק Wikipedia

| | |
|---|---|
| **תלונה** | GitHub / מידע כללי — רוב providers נכשלים, רק Wikipedia מוצג |
| **שורש הבעיה** | שילוב: intent routing שגוי, query מלוכלך, structured fail בלי fallback, dedup חסר (GitHub + Wiki EN + HE במקביל) |
| **מסקנה** | לא באג יחיד — שרשרת של חילוץ + routing + הצפת context |
| **שינוי** | intent dedup · query cleaning · provider-specific extractors · UI שמבדיל structured vs wiki |

---

### 1.4 קריסת WebGPU — שאלת GitHub

| | |
|---|---|
| **תלונה** | `GatherBlockQuantized` / `Integer overflow` ב-OrtRun אחרי "פרויקטים הפופולarיים ב-GitHub השבוע" |
| **שגיאות** | WebGPU → retry WASM עם **אותו prompt ענק** → שגיאות נוספות |
| **שורש הבעיה** | **לא חומרה** (32GB RAM מספיק). prompt כולל: history (14k chars) + webContext **ללא cap** + system + `maxNewTokens` 2048. GitHub + Wikipedia EN+HE = אלפי תווים נוספים |
| **קובץ** | `orchestrator.ts`, `model.worker.ts`, `chatIntents.ts`, `App.tsx` |
| **מסקנה** | תקציב prompt **כולל** חסר; retry WASM לא מקצץ context |
| **שינוי** | `chatResourceBudget.ts` · cap על webContext · `maxNewTokens` דינמי (search=384, code=1536) · truncate לפני retry · GitHub בלי Wikipedia כפול |

---

### 1.5 אין "אוויר בשיחה" / הגדרות לפי חומרה

| | |
|---|---|
| **תלונה** | (הצעה בשיחה) — לדעת כמה context נשאר; presets ל-32GB+8GB VRAM vs 16GB+8GB |
| **מצב היום** | `CHAT_HISTORY_CHAR_BUDGET=14_000` — **רק history**. `detectVisionBudget()` — **רק vision**. `maxNewTokens` default 2048, max UI 4096 |
| **מסקנה** | המשתמש לא רואה מתי קרוב ל-overflow; הגדרות לא מותאמות ל-Gemma 4B בדפדפן |
| **שינוי** | StaminaBar + פרופילים (Ultra/Balanced/Safe/Low) + המלצות maxNewTokens |

---

### 1.6 חיפוש מציף את המודל — אין סינון חכם

| | |
|---|---|
| **תלונה** | איך לשלוף מידע מסודר בלי להציף? side thread → סיכום → links → מחיקת raw? |
| **מצב היום** | `formatWebContext()` שולח **טקסט מלא** ל-system message. UI (`SearchSourcesBlock`) כבר מציג raw — אבל המודל מקבל הכל |
| **מה טוב** | raw **לא** נשמר ב-`grovee_chats_v1` — רק ב-turn הנוכחי |
| **מסקנה** | דפוס sub-agent / SearchBrief — industry standard (Anthropic, Perplexity) |
| **שינוי** | `buildSearchBrief()` rules-based (P1) + side LLM compress אופציונלי (P2) |

---

### 1.7 ראייה + worker mutex (מתוך ניתוח ראשוני)

| | |
|---|---|
| **תלונה** | תהליכי ראייה צריכים לרוץ רק עם מצלמה; לא לתקוע צ'אט |
| **מצב היום** | `pauseForChatInference()` — no-op. YOLO רץ גם בזמן chat. worker יחיד: `chatBusy`/`sceneBusy` — chat מחכה עד 180s ל-`analyze_scene` |
| **מסקנה** | contention על worker + עומס GPU/CPU |
| **שינוי** | P2: pause vision בזמן generate · הקטנת timeout scene · `syncVisionBusy` שעוצר בפועל |

---

### 1.8 רעש Chrome extension

| | |
|---|---|
| **תלונה** | `message channel closed before a response was received` |
| **מסקנה** | extension של דפדפן — **לא** באג GROVEEMODEL |
| **שינוי** | אין — אופציונלי: הערה ב-TROUBLESHOOTING |

---

## 2. ארכיטקטורת יעד (מצב רצוי)

```
שאלת משתמש
    ↓
sanitizeQuery + classifyIntents (dedup providers)
    ↓
runWebSearch() — parallel fetch
    ↓
buildSearchBrief() — ≤800 תווים + links
    │                              ↓
    │                    SearchSourcesBlock (UI — raw מלא)
    ↓
estimatePromptBudget(history + brief + system + images)
    ↓
[אם over budget] trim history / cap brief / reduce maxNewTokens
    ↓
GlobePanel — רק אם isGlobePresentationQuery / map request
    ↓
Gemma generate (maxNewTokens דינמי)
    ↓
[retry WASM] — עם context מקוצץ, לא אותו prompt
    ↓
תשובה + sources ב-UI (ללא raw ב-history)
```

---

## 3. תוכנית עבודה לפי שלבים

---

### שלב P0 — יציבות קריטית (לפני הכל)

**מטרה:** למנוע קריסות overflow ותיקון באגי חיפוש בסיסיים.

#### P0.1 — תקציב Prompt כולל

- [ ] קובץ חדש: `app/src/chatResourceBudget.ts`
- [ ] `estimatePromptChars({ history, webBrief, systemPrompt, imageCount })`
- [ ] `trimToBudget()` — history קודם, אחר כך web, אחר כך maxNewTokens
- [ ] קבועים: `TOTAL_PROMPT_CHAR_BUDGET` (~18k–22k לפי פרופיל), `WEB_BRIEF_CHAR_BUDGET` (600–800)
- [ ] אינטגרציה ב-`App.tsx` לפני `postMessage generate`
- [ ] unit tests

#### P0.2 — SearchBrief (דחיסה rules-based)

- [ ] קובץ: `app/src/webSearch/searchBrief.ts`
- [ ] `buildSearchBrief(sources, intents, query): SearchBrief`
- [ ] per-provider formatters: weather, github, worldtime, wikipedia (2–3 משפטים), country, earthquake
- [ ] cap: עד 8 facts, 6 links, 600–800 תווים
- [ ] `orchestrator.ts`: `contextText` = brief, לא raw
- [ ] `SearchSourcesBlock` — נשאר raw (ללא שינוי UX)
- [ ] unit tests

#### P0.3 — Intent dedup

- [ ] `intents.ts`: GitHub query → **לא** Wikipedia
- [ ] structured intents (worldtime, weather…) → wiki רק ב-fallback explicit
- [ ] `orchestrator.ts`: fallback Wikipedia כש-structured נכשל (לא חוסם)
- [ ] tests לשאלות: GitHub-only, time-only

#### P0.4 — תיקון חילוץ מיקום / שעון

- [ ] `sanitizeSearchQuery()` — הסר ברכות, "מר גrובi", סימני שאלה
- [ ] `queryExtract.ts`: regex `בישrael`, `בישrael`, `ישrael` ללא רווח חובה
- [ ] `worldTime.ts`: fallback `Israel` / `Asia/Jerusalem` לפני geocode כללי
- [ ] **לא** fallback ל-`query.trim()` כש-chunk > 40 תווים
- [ ] tests: "בוקר טוב… מה השעה בישrael"

#### P0.5 — Globe gating

- [ ] `shouldOpenGlobePanel()`: intent בלבד ≠ true
- [ ] פתיחה רק: `isGlobePresentationQuery()` OR explicit map regex OR place + presentation
- [ ] weather/time/country **בלי** Globe אוטומטי
- [ ] tests ב-`realityGlobe/intents.test.ts`

#### P0.6 — maxNewTokens דינמי + retry בטוח

- [ ] `App.tsx`: search turn → 384–512; code → 1536; default → 768 (Ultra) / 512 (Balanced)
- [ ] `model.worker.ts`: WASM retry **אחרי** truncate webContext/history
- [ ] `isWebGpuRuntimeError`: overflow → truncate, לא reload אותו prompt
- [ ] test/manual: GitHub אחרי שיחה ארוכה

**קריטריון סיום P0:**

- [ ] שאלת GitHub אחרי 20 turns — **לא** קורס
- [ ] "מה השעה בישrael" — TimeAPI OK
- [ ] "מה מזג האוויר בתל אביב" — **לא** פותח Globe
- [ ] vitest ירוק

---

### שלב P1 — UX "אוויר בשיחה" + הגדרות חכמות

**מטרה:** שקיפות למשתמש + presets לפי חומרה.

#### P1.1 — פרופילי חומרה

- [ ] `app/src/chatHardwareProfile.ts`
- [ ] זיהוי: `navigator.deviceMemory`, `hardwareConcurrency`, WebGPU (heuristic)
- [ ] פרופילים: Ultra (32GB+), Balanced (16GB), Safe, Low
- [ ] mapping: budgets + default maxNewTokens + inference device hint

#### P1.2 — Conversation Stamina Bar

- [ ] קומponent: `ConversationStaminaBar.tsx`
- [ ] חישוב: `used / budget` מ-`estimatePromptChars` + הודעה אחרונה
- [ ] צבעים: ירוק >60%, צהוב 30–60%, אדום <30%
- [ ] tooltip: "היסטוריה X · חיפוש Y · מקס tokens Z"
- [ ] CSS ב-`App.css`

#### P1.3 — Settings presets

- [ ] `SettingsModal`: בחירת פרופיל + "המלצה אוטומטית"
- [ ] ברירות: Ultra maxNewTokens 768, Balanced 512, Safe 384
- [ ] אזהרה כש-maxNewTokens > 1024 על Balanced
- [ ] שמירה ב-localStorage (`grovee_chat_profile_v1`)

#### P1.4 — Fast-path (אופציונלי ב-P1)

- [ ] שאלות pure structured (weather, time) → תשובה ישירה **בלי** Gemma
- [ ] חוסך tokens + latency

**קריטריון סיום P1:**

- [ ] Stamina bar מתעדכן בכל turn
- [ ] preset Ultra/Balanced נשמר ומשפיע על budgets

---

### שלב P2 — חיפוש מתקדם + Side compress

**מטרה:** שאלות multi-source מורכבות בלי הצפה.

#### P2.1 — Side LLM compress (אופציונלי)

- [ ] `compressSearchWithModel()` — turn נפרד, maxNewTokens 256
- [ ] רץ רק אם `buildSearchBrief` rules > budget
- [ ] **לא** נשמר ב-history
- [ ] mutex: לפני main generate

#### P2.2 — Cache חיפוש

- [ ] cache 5 דקות לפי `(intent + normalizedQuery)`
- [ ] `SEARCH_PLAN.md` Phase 4.5

#### P2.3 — Fast-path Knowledge Graph פנימי

- [ ] cache מדינה → PM, בירה, מטבע (Phase 4.9)

**קריטריון סיום P2:**

- [ ] שאלה multi-source → brief ≤800 תווים תמיד
- [ ] side compress רק when needed

---

### שלב P3 — ראייה ו-worker (יציבות ארוכת טווח)

**מטרה:** מצלמה לא מתנגשת עם chat inference.

#### P3.1 — Vision pause בזמן generate

- [ ] `GroveeVisionRunner.pauseForChatInference()` — pause בפועל
- [ ] `App.tsx` `syncVisionBusy` — hold/resume סביב generate
- [ ] vision רק כש-`cameraActive`

#### P3.2 — Worker contention

- [ ] review timeout 180s ל-`analyze_scene`
- [ ] priority: chat > scene כש-generating
- [ ] או: scene analysis ב-worker נפרד (עתידי)

#### P3.3 — בדיקות regression

- [ ] `vision-always-on.test.ts` — עדכון לציפיות חדשות
- [ ] camera OFF → zero vision CPU

**קריטריון סיום P3:**

- [ ] chat + camera ON — generate לא נתקע 180s
- [ ] vision loops רק עם מצלמה

---

### שלב P4 — בדיקות Acceptance (Regression Suite)

| # | שאלה / פעולה | צפוי |
|---|--------------|------|
| A1 | "בוקר טוב מר גrובi, איזhe יום ומה השעha בישrael" | TimeAPI OK, תשובה עם שעה/יום |
| A2 | "מה מזג האוויר בניו יורק?" | Open-Meteo, **Globe סגור** |
| A3 | "הצג על המפה את berlin" | Globe **נפתח** |
| A4 | "מהם הפרויקטים הפופולarיים ב-GitHub השבוע?" | GitHub OK, **ללא** Wikipedia, **ללא** crash |
| A5 | שיחה 15+ turns + A4 | Stamina אדום/צהוב, **עדיין** לא crash |
| A6 | "חפש מידע על פירamides" | Wikipedia בלבד |
| A7 | Search ON + קובץ מצורף | חיפוש **לא** רץ |
| A8 | מצלמה OFF | אין vision loops |
| A9 | מצלמה ON + שאלת text | chat תוך <30s (לא 180s wait) |
| A10 | Settings → WASM אחרי WebGPU fail | retry עם context מקוצץ |

- [ ] כל 10 הבדיקות עוברות ידנית
- [ ] vitest: intents, searchBrief, queryExtract, globe, chatResourceBudget

---

## 4. קבצים מרכזיים לעריכה

| קובץ | שלב | שינוי |
|------|-----|-------|
| `app/src/chatResourceBudget.ts` | P0 | **חדש** |
| `app/src/webSearch/searchBrief.ts` | P0 | **חדש** |
| `app/src/webSearch/orchestrator.ts` | P0 | brief במקום raw |
| `app/src/webSearch/intents.ts` | P0 | dedup, fallback |
| `app/src/webSearch/queryExtract.ts` | P0 | regex עברית |
| `app/src/webSearch/providers/worldTime.ts` | P0 | sanitization + fallback |
| `app/src/realityGlobe/intents.ts` | P0 | Globe gating |
| `app/src/App.tsx` | P0–P1 | budgets, stamina, dynamic tokens |
| `app/src/model.worker.ts` | P0 | truncate before retry |
| `app/src/chatIntents.ts` | P0 | integrate total budget |
| `app/src/chatHardwareProfile.ts` | P1 | **חדש** |
| `app/src/ConversationStaminaBar.tsx` | P1 | **חדש** |
| `app/src/GroveeVisionRunner.ts` | P3 | pause vision |
| `app/src/webSearch/SEARCH_PLAN.md` | P2 | עדכון ארכיטקטורה |

---

## 5. סדר ביצוע מומלץ (יום-עבודה)

```
יום 1: P0.4 (time) + P0.5 (Globe) + tests
יום 2: P0.2 (SearchBrief) + P0.3 (dedup)
יום 3: P0.1 (budget) + P0.6 (tokens + retry)
יום 4: P1 (Stamina + profiles)
יום 5: P4 acceptance + P3 vision (אם נשאר זמן)
יום 6+: P2 advanced search
```

---

## 6. מה **לא** לעשות

- [ ] לא לשמור `webContext` / raw search ב-`localStorage` history
- [ ] לא לשלוח raw + brief — כפילות
- [ ] לא Wikipedia + GitHub על אותה שאלה structured
- [ ] לא WASM retry עם prompt מלא אחרי overflow
- [ ] לא לפתוח Globe על weather/time/country בלי בקשת מפה
- [ ] לא commit credentials / secrets

---

## 7. צ'ק-ליסט מאסטר (סימון סופי)

### באגים שזוהו

- [ ] Globe נפתח בכל intent — **תוקן**
- [ ] TimeAPI / `בישrael` — **תוקן**
- [ ] GitHub overflow — **תוקן**
- [ ] Wikipedia בלבד / providers נכשלים — **תוקן**
- [ ] webContext ללא cap — **תוקן**
- [ ] maxNewTokens קבוע 2048 — **תוקן**
- [ ] WASM retry ללא truncate — **תוקן**
- [ ] Vision רץ בזמן chat — **תוקן** (P3)
- [ ] אין Stamina UI — **תוקן** (P1)
- [ ] אין SearchBrief — **תוקן** (P0)

### תשתית חדשה

- [ ] `chatResourceBudget.ts`
- [ ] `searchBrief.ts`
- [ ] `chatHardwareProfile.ts`
- [ ] `ConversationStaminaBar.tsx`
- [ ] tests לכל module חדש

### Acceptance

- [ ] A1–A10 עברו
- [ ] vitest ירוק
- [ ] `SEARCH_PLAN.md` עודכן

---

## 8. סיכום מנהלים

| בעיה | סיבה | פתרון מרכזי | עדיפות |
|------|------|-------------|--------|
| Globe בכל שאלה | intent = open map | gating לפי בקשת מפה | P0 |
| שעה בישrael | query extract + no fallback | sanitize + regex + Israel TZ | P0 |
| קריסת GitHub | prompt overflow | SearchBrief + total budget | P0 |
| חיפוש מציף | formatWebContext raw | buildSearchBrief | P0 |
| אין feedback | אין budget UI | StaminaBar + profiles | P1 |
| vision תוקע | worker mutex / no pause | pause + priority | P3 |

---

## 9. קבצים קיימים רלוונטיים (מצב נוכחי)

| קובץ | תפקיד |
|------|--------|
| `app/src/model.worker.ts` | load/generate, mutex, WebGPU→WASM fallback |
| `app/src/webSearch/orchestrator.ts` | `runWebSearch`, `formatWebContext` |
| `app/src/webSearch/intents.ts` | routing, `buildGitHubSearchQuery`, `shouldOpenGlobePanel` |
| `app/src/webSearch/queryExtract.ts` | `extractLocationPhrase` |
| `app/src/webSearch/providers/worldTime.ts` | TimeAPI + geocoding |
| `app/src/realityGlobe/intents.ts` | Globe panel logic |
| `app/src/chatIntents.ts` | `trimHistoryForContext`, `CHAT_HISTORY_CHAR_BUDGET` |
| `app/src/visionBudget.ts` | `detectVisionBudget` (vision only) |
| `app/src/characterPrompts.ts` | `WEB_SEARCH_GROUNDING_APPEND` |
| `app/src/SearchSourcesBlock.tsx` | UI מקורות (raw) |
| `app/src/webSearch/SEARCH_PLAN.md` | תוכנית חיפוש קיימת |

---

*מסמך זה הוא מקור האמת לתוכנית העבודה. עדכן צ'ק-ליסטים ככל שלב מושלם.*

---

## 10. QA מלא + מקורות מידע (נוסף 2026-06-13)

### קובץ acceptance
- `app/src/webSearch/acceptanceQueries.ts` — `ACCEPTANCE_QUERIES`, `QA_UNSUPPORTED_QUERIES`, `DATA_SOURCE_REGISTRY`

### בדיקות ידניות (שלח בצ'אט — חיפוש אוטומטי)

| קטגוריה | שאלות |
|---------|--------|
| בסיסיות | מטבע ברזיל · בירת יפן · PM בריטניה · אוכלוסיית קנדה |
| זמן | שעה בטוקיו · תאריך NY · הפרש ישראל-לונדון · בוקר טוב + שעה בישrael |
| מזג אוויר | טמפרatura ת"א · גשם לונדון · רוח פריז (**Globe סגור**) |
| מטבעות | 100 USD→ILS · 1 USD→BRL · 1000 ILS→EUR |
| מפות | בית חולים ליד אייפel · דלק ליד הית'רo · רכבת ליד הית'רo · km י-ם-חיפה |
| GitHub | פופולריים השבוע · WebGPU (**ללא crash**) |
| HF | מודלי תמונה פופולarיים |
| קריפטו | מחיר BTC |
| רעידות / טיסות / ISS | USGS · מטוסים מעל ישrael · ISS מעל ישrael |

### צפוי unsupported (הודעה ברורה, לא המצאה)
Reddit · מניות NVIDIA/S&P · זהב · ספינות בנמל חיפה

### מקורות — סטטוס

| מקור | סטטוס |
|------|--------|
| Wikipedia, Wikidata, Open-Meteo, OSM/Nominatim, TimeAPI, Frankfurter, REST Countries, GitHub, HF, USGS, CoinGecko, Hacker News | ✅ live |
| News RSS, ADSB, ISS, GDACS | ⚠️ partial (CORS/rate) |
| Reddit, Finnhub, Alpha Vantage, Brave, SearXNG, Overpass, arXiv | 📋 planned / needs-key |

### UI חיפוש (Perplexity-style)
- `SearchProgressPanel` — "מחפש ברשת", רשימת providers בזמן אמת, סיכום brief, raw expand
- `ConversationStaminaBar` — **זמני** — יוחלף ב-**Context Ring** (ראה `IMPROVEMENT_WORKPLAN.md` §3)
- הגדרות → פרופיל Ultra/Balanced/Safe/Low

### יושם בקוד (2026-06-13)
- [x] P0: `chatResourceBudget.ts`, `searchBrief.ts`, intent dedup, sanitize query, Globe gating, WASM retry truncate
- [x] P1: `chatHardwareProfile.ts`, StaminaBar, settings presets
- [x] P3 חלקי: vision pause בזמן chat generate
- [x] providers: CoinGecko, Hacker News, unsupported stubs
- [ ] P2: Side LLM compress, cache 5min, arXiv, Overpass, SearXNG
