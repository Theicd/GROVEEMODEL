# GROVEEMODEL — תוכנית שיפור מוצר (Product Improvement Plan)

**פרויקט:** `c:\Users\Avatar001\CascadeProjects\GROVEEMODEL`  
**עדכון:** 2026-06-13  
**מטרה:** ממשק ברמת מוצר בינלאומית — יציב, ברור, יעיל בטוקנים, חיפוש חכם, זיכרון מנוהל  
**מסמכים קשורים:** `WORK_PLAN.md` (היסטוריה + QA), `app/src/webSearch/acceptanceQueries.ts`  
**מקור השראה טכני:** `c:\BRAIN2\clouad\src` (Continue/Claude Code patterns)

---

## 0. עקרונות מוצר (חובה בכל שלב)

| עקרון | משמעות |
|--------|---------|
| **Clarity first** | המשתמש מבין תוך 3 שניות מה קורה (טוען / מחפש / חושב / נגמר context) |
| **Progressive disclosure** | מידע מפורט (טוקנים, מקורות, breakdown) — **בלחיצה**, לא תמיד על המסך |
| **No prompt flooding** | raw search / vision / history לא נכנסים למודל בלי cap + brief |
| **Honest failures** | "מקור לא מחובר" — לא המצאת מספרים |
| **RTL + EN** | עברית RTL, מספרים/URLs LTR, accessibility (ARIA, keyboard) |
| **Local-first** | Gemma 4B בדפדפן — כל feature נמדד ב-latency ו-RAM |

**רפרנס UX (לא להעתיק 1:1):** Cursor context ring · ChatGPT usage popover · Perplexity search steps · Claude Code compact warning

---

## 1. מצב נוכחי vs יעד

| תחום | היום (יושם) | יעד מוצר |
|------|-------------|----------|
| תקציב prompt | `chatResourceBudget`, SearchBrief | + breakdown לפי סוג + thresholds warning/block |
| מד "אוויר" | `ConversationStaminaBar` — **פס אופקי** | **Context Ring** — עיגול + popover (ראה §3) |
| חיפוש UI | `SearchProgressPanel` — steps + sources | Perplexity-grade: live steps, collapsed history |
| זיכרון | camera memory, localStorage chats | session summary + selective recall (clouad pattern) |
| compaction | trim history בלבד | microCompact + optional side-summarize |
| providers | ~20 live | + WebFetch, Overpass, arXiv; stubs ברורים ל-API key |
| QA | `acceptanceQueries.ts` 40+ | automated live tests + manual checklist |

---

## 2. ארכיטектורת יעד (זרימה)

```
User message
    ↓
Intent router (dedup, sanitize) ──→ SearchProgressPanel (UI only)
    ↓
Providers parallel fetch
    ↓
buildSearchBrief() ──→ model (≤800 chars)
    ↓
prepareChatContext() ──→ Context Ring % (remaining context)
    ↓
Gemma generate (dynamic maxNewTokens)
    ↓
[overflow] truncate + WASM retry (already partial)
    ↓
Optional: session memory update (side thread, not in history)
```

---

## 3. UX — Context Ring (החלפת Stamina Bar) — **עדיפות P1-UX**

> **בעיה:** פס "אוויר X%" לא תואם ממשקי AI/IDE מובילים ונראה טכני מדי.  
> **פתרון:** עיגול קטן ליד composer / בסרגל עליון — לחיצה פותחת פanel.

### 3.1 מיקום — ✅ בוצע (2026-06-13)

- [x] **Desktop:** עיגול 22px + אחוז בשורת `composer-modes` (ליד Think/Camera)
- [ ] **Mobile:** popover full-width bottom sheet (כרגע popover רגיל)
- [ ] **Camera mode:** עיגול נפרד או מאוחד עם badge "vision" — לא להסתיר HAL status

### 3.2 מראה (Context Ring) — ✅ בוצע

- [x] SVG ring — `stroke-dasharray` לפי **אחוז נותר**
- [x] צבעים: ירוק >50% · כתום 20–50% · אדום <20%
- [ ] pulse אדום <10%
- [x] אחוז מוצג ליד הטבעת
- [x] Tooltip on hover
- [x] **לא** טקסט "אוויר" — הוסר לגמרי

### 3.3 Popover / Panel (לחיצה על העיגול)

| שדה | תוכן |
|-----|------|
| **Context remaining** | X% · ~Y / Z chars (או ~tokens estimated) |
| **Breakdown** | History · Search brief · System · Images · Last turn |
| **Profile** | Ultra / Balanced / Safe / Low + link להגדרות |
| **Actions** | "Compact conversation" (עתידי) · "New chat" · "Clear search cache" |
| **Warning** | כש-<20%: "שיחה ארוכה — שאלות קצרות או chat חדש" |

### 3.4 נתונים (מ-`PreparedChatContext`) — ✅ בוצע

- [x] `prepareChatContext` כבר מחזיר `breakdown`, `totalBudget`, `usedChars`
- [x] `remainingPercent = staminaPercent`
- [x] `estimateTokens(chars) = chars / 4` לתצוגה ב-popover

### 3.5 קבצים ליישום — ✅ בוצע

| קובץ | פעולה | סטטוס |
|------|--------|--------|
| `app/src/ContextRing.tsx` | **חדש** — ring + popover | ✅ |
| `app/src/ConversationStaminaBar.tsx` | **נמחק** | ✅ |
| `app/src/SearchProgressPanel.tsx` | duplicate `ConversationStaminaBar` הוסר | ✅ |
| `app/src/App.tsx` | wire `ContextRing` + `contextUsage` state | ✅ |
| `app/src/index.css` | `.context-ring`, `.context-popover` | ✅ |

### 3.6 צ'ק-ליסט UX Context Ring

- [x] Ring מוצג רק כש-`isLoaded && messages.length > 0`
- [x] מתעדכן אחרי כל `prepareChatContext`
- [x] Popover נסגר ב-Escape / click outside
- [x] RTL: popover מיושר נכון (dir="rtl", inset-inline-start)
- [x] a11y: `aria-expanded`, `aria-label`, `role="dialog"`
- [ ] אין regression ל-composer height במobile — לבדוק ידנית

---

## 4. שלב A — יציבות ומנוע (השלמת P0/P1)

**מטרה:** zero crash על שיחה ארוכה + search; thresholds ברורים.

### A.1 Context engine (מ-clouad: `autoCompact.ts`, `contextAnalysis.ts`)

- [ ] `contextBreakdown.ts` — פירוק כמו `analyzeContext`: history / web / system / images / vision
- [ ] Warning threshold: `<20%` remaining → banner עדין (לא modal)
- [ ] Block threshold: `<5%` → cap אגרסיבי + הודעה לפני generate
- [ ] Circuit breaker: 2× overflow → force profile Safe + maxNewTokens 256
- [ ] unit tests

### A.2 Search pipeline (השלמה)

- [ ] `WebFetchTool` pattern → `providers/webFetch.ts` (URL → extract ≤500 chars)
- [ ] Side compress (P2): `compressSearchWithModel()` — template מ-clouad `compact/prompt.ts` סעיפים 1–5
- [ ] Cache 5min: `searchCache.ts` keyed by `(intent + normalizedQuery)`
- [x] Wikipedia fallback — **תוקן (2026-06-13):** אין fallback לוויקיפדיה עבור intents של live-data בלבד (currency/weather/time/crypto/aviation…) — ויקיפדיה לא יכולה לענות עליהם
- [x] **תיקוני ניתוב (2026-06-13):** "כמה יורו שווים 1000 שקלים" → currency עם amount; "כמה תושבים יש בקנדה" → country; "מהם 10 הנושאים…" לא נופל ל-aviation; "מה בירת יפן" → country; "מה התאריך ב…" → worldtime; "מה הטמפרטורה/מהירות הרוח" → weather; "תחנת החלל" → satellite; crypto ללא wikipedia
- [x] **Frankfurter:** תמיכה בסכום ("1000 ILS = X EUR") + זיהוי alias עם סימן שאלה צמוד

### A.3 Vision + worker

- [ ] `pauseForChatInference` — verify pipeline נעצר (לא רק onUiTick)
- [ ] cap `visionContext` in prompt (microCompact pattern)
- [ ] scene wait: הורד 180s → 30s או skip scene if chat pending
- [ ] vision רק כש-`cameraActive`

### A.4 צ'ק-ליסט שלב A

- [ ] GitHub after 20 turns — no crash
- [ ] Weather — Globe stays closed
- [ ] "בוקר טוב + שעה בישrael" — TimeAPI OK
- [x] vitest green (222 unit + 50/51 live; כשל יחיד = timeout רשת של open-meteo)
- [x] `npm run build` green
- [x] **Watchdog (2026-06-13):** אם ה-worker משתתק >180ש' באמצע יצירה — הצ'אט משוחרר אוטומטית עם הודעת שגיאה ברורה במקום להיתקע ולהתעלם מהודעות חדשות
- [x] **Worker hardening:** `buildInputs` בתוך try/catch — חריגה בהכנת prompt לא משאירה `chatBusy=true` לנצח
- [x] **Feedback בשליחה כפולה:** שליחה בזמן יצירה מציגה "עדיין עונה — המתן או עצור" במקום התעלמות שקטה

---

## 5. שלב B — זיכרון ו-compaction (מ-clouad)

**מקור:** `memdir/`, `SessionMemory/`, `microCompact.ts`, `sideQuery.ts`

### B.1 Session memory (rolling summary)

- [ ] `sessionMemory.ts` — summary file per chat session (לא ב-prompt כל turn)
- [ ] Thresholds (clouad defaults): init 10k · update +5k · every 3 search turns
- [ ] inject רק 300–500 chars summary + "last user goals"

### B.2 Selective recall

- [ ] `findRelevantMemorySnippets(query)` — עד 3 snippets מ-`MEMORY.md` / daily notes
- [ ] **לא** inject full memory files

### B.3 Micro-compact (ללא LLM)

- [ ] Strip old `webContext` from worker message (already ephemeral)
- [ ] Trim vision blocks older than 2 turns from history metadata
- [ ] Replace duplicate search raw in UI with "🔍 3 sources (collapsed)"

### B.4 Manual / auto compact (עתידי)

- [ ] כפתור ב-Context Ring popover: "Summarize & continue"
- [ ] Side thread Gemma 512 tokens → boundary message → history replaced

### B.5 צ'ק-ליסט שלב B

- [ ] 30-turn chat — ring stable, no overflow
- [ ] Memory snippet רלוונטי לשאלה, לא spam
- [ ] Compact לא מוחק search links מה-UI

---

## 6. שלב C — חיפוש ומקורות (מוצר + QA)

### C.1 Providers חדשים

| מקור | עדיפות | הערות |
|------|--------|--------|
| WebFetch (post-search) | P1 | extract page text |
| Overpass/OSM POI | P2 | hospitals, stations |
| arXiv | P3 | papers |
| Brave/SearXNG | P3 | self-host / API key |
| Finnhub / Alpha Vantage | P3 | stocks — needs key |
| Reddit | P3 | OAuth stub |

### C.2 Search UI (Perplexity-grade)

- [ ] שלב "מנתב שאילתה…" → per-provider rows עם spinner → ✓/✗
- [ ] Collapse completed search in message history (clouad `collapseReadSearch`)
- [ ] Links כרטיסיות (favicon + title + domain)
- [ ] Search brief preview — 3 bullets max in collapsed state

### C.3 QA

- [ ] הרץ `acceptanceQueries.ts` — כל `ACCEPTANCE_QUERIES`
- [ ] `QA_UNSUPPORTED_QUERIES` — מודל אומר "לא מחובר" בעברית
- [ ] `npm run test:live` (אם קיים) — network tests
- [ ] רשום תוצאות ב-`QA_RESULTS.md` (תאריך + pass/fail)

### C.4 צ'ק-ליסט שלב C

- [ ] 40+ acceptance queries documented
- [ ] DATA_SOURCE_REGISTRY מעודכן
- [ ] אין Wikipedia על GitHub-only queries

---

## 7. שלב D — עיצוב מוצר בינלאומי (Design System)

### D.1 Composer area

- [ ] Context Ring במקום stamina bar
- [ ] Status line: "מחפש…" / "חושב…" — icon + verb (clouad `spinnerVerbs`)
- [ ] Attachments preview — consistent chips
- [ ] Empty state landing — `ChatLandingHero` polished

### D.2 Messages

- [ ] Search block — collapsed by default after reply
- [ ] Code blocks — copy button
- [ ] Error states — actionable (WASM hint, clear cache)

### D.3 Settings

- [ ] Hardware profile cards עם icons
- [ ] Inference backend (WebGPU/WASM) — warning copy ברור
- [ ] Search toggle removed? (auto search) — אם כן, הסבר ב-onboarding

### D.4 Onboarding (חד-פעמי)

- [ ] 3 slides: Load model · Context ring · Search sources
- [ ] Skip + don't show again

### D.5 צ'ק-ליסט Design

- [ ] Dark theme consistent (tokens in CSS variables)
- [ ] Font sizes: body ≥14px, labels ≥12px
- [ ] Touch targets ≥44px mobile
- [ ] Lighthouse accessibility ≥90 on chat page

---

## 8. מפת clouad → GROVEEMODEL (יישום)

| clouad (`BRAIN2/clouad/src`) | יישום ב-GROVEEMODEL |
|------------------------------|---------------------|
| `services/compact/autoCompact.ts` | thresholds + circuit breaker (§A.1) |
| `services/compact/microCompact.ts` | vision/search strip (§B.3) |
| `services/compact/prompt.ts` | side compress template (§A.2) |
| `utils/sideQuery.ts` | `compressSearchWithModel` (§A.2) |
| `utils/contextAnalysis.ts` | `contextBreakdown.ts` (§A.1) |
| `components/TokenWarning.tsx` | Context Ring popover warnings (§3) |
| `utils/collapseReadSearch.ts` | collapsed search UI (§C.2) |
| `memdir/findRelevantMemories.ts` | selective memory (§B.2) |
| `services/SessionMemory/*` | session rolling summary (§B.1) |
| `constants/toolLimits.ts` | caps on raw provider text (§A.2) |
| `tools/WebFetchTool/` | `webFetch.ts` provider (§C.1) |
| `CONTINUE_AGENT_WORKPLAN.md` | סדר עדיפויות §2–7 |

---

## 9. סדר ביצוע מומלץ (sprints)

| Sprint | משך | תוכן | Definition of Done |
|--------|-----|------|-------------------|
| **S1** | 2–3 ימים | Context Ring + popover (§3) + הסר stamina bar | UX review OK |
| **S2** | 2 ימים | contextBreakdown + warnings (§A.1) | Ring shows real breakdown |
| **S3** | 2 ימים | Search UI polish + collapse (§C.2) | Perplexity-like flow |
| **S4** | 2 ימים | WebFetch + cache (§A.2, C.1) | QA queries pass |
| **S5** | 3 ימים | Session memory + selective recall (§B) | 30-turn stable |
| **S6** | 2 ימים | Design pass + onboarding (§D) | Lighthouse a11y |
| **S7** | 1 יום | Full QA + `QA_RESULTS.md` | All acceptance |

---

## 10. צ'ק-ליסט מאסטר (סימון סופי)

### UX / מוצר
- [x] Context Ring (עיגול) במקום פס stamina — 2026-06-13
- [x] Popover: breakdown + tokens + profile + warning
- [x] Search progress — live providers
- [ ] Collapsed search in history
- [ ] Warning banner <20% context (כרגע רק ב-popover)
- [ ] Onboarding 3 slides
- [x] RTL + a11y בסיסי ל-Context Ring

### מנוע
- [x] SearchBrief cap enforced
- [x] prepareChatContext total budget
- [x] Dynamic maxNewTokens
- [x] WASM retry truncate
- [x] Generation watchdog — שחרור צ'אט תקוע (180s) — 2026-06-13
- [ ] Circuit breaker overflow
- [ ] contextBreakdown module (standalone)
- [x] Globe gating verified
- [x] Time/Israel query verified

### זיכרון
- [ ] Session rolling summary
- [ ] Selective memory snippets
- [ ] microCompact vision/search UI
- [ ] Optional manual compact

### חיפוש / מקורות
- [x] Intent dedup + תיקוני ניתוב QA (currency/country/aviation/satellite/weather/date) — 2026-06-13
- [x] Wikipedia fallback rules — אין fallback ל-live-data intents
- [ ] WebFetch provider
- [ ] Search cache 5min
- [x] Unsupported stubs (Reddit, stocks)
- [x] DATA_SOURCE_REGISTRY updated

### QA
- [x] vitest unit 100% pass (222)
- [x] live acceptance 50/51 (כשל יחיד: timeout רשת open-meteo)
- [x] build pass
- [ ] ACCEPTANCE_QUERIES manual pass בדפדפן
- [ ] QA_UNSUPPORTED honest messages
- [ ] No WebGPU overflow on long chat + GitHub (ידני)

### תיעוד
- [ ] WORK_PLAN.md status updated
- [x] IMPROVEMENT_WORKPLAN.md checklists marked
- [ ] QA_RESULTS.md with date

---

## 11. מה לא לעשות

- [ ] לא stamina bar אופקי ב-production
- [ ] לא raw search ב-history/localStorage
- [ ] לא Globe on weather/time
- [ ] לא duplicate Wikipedia + GitHub
- [ ] לא WASM retry without truncate
- [ ] לא inject full memory files each turn
- [ ] לא UI noise — כל indicator חייב click-to-expand

---

## 12. קבצים קיימים (נקודות עיגון)

| קובץ | תפקיד |
|------|--------|
| `app/src/chatResourceBudget.ts` | תקציב + staminaPercent |
| `app/src/chatHardwareProfile.ts` | Ultra/Balanced/Safe/Low |
| `app/src/webSearch/searchBrief.ts` | דחיסה למודל |
| `app/src/SearchProgressPanel.tsx` | UI חיפוש |
| `app/src/ConversationStaminaBar.tsx` | **להחליף** → ContextRing |
| `app/src/webSearch/acceptanceQueries.ts` | QA registry |
| `WORK_PLAN.md` | היסטוריה + באגים מקוריים |

---

*עדכן צ'ק-ליסט §10 אחרי כל sprint. מסמך זה הוא מקור האמת לשיפורי מוצר מ-2026-06-13.*
