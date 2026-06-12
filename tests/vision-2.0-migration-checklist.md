# Vision 2.0 — צ'קליסט מיגרציה + QA

> **סטטוס יישום (2026-06-10):** Phases 0–7 **Complete**. Feature flag: `vision.vision2Enabled` (ברירת מחדל: **ON**). Rollback: כבה ב-⚙ → עיניים ואסיטואציות.

מסמך זה מלווה את המעבר מארכיטקטורת **Sensors → Events → Gemma** ל-**Sensors → Perception → Understanding → Reasoning → Personality → Dialogue**.

**עקרון QA קבוע:** בכל שלב — לפני sign-off — חובה לוודא ש**ממשק החיישנים** (Vision Inspector, overlay, FPS, toggles) **לא נפגע**.

---

## לפני שמתחילים

| # | משימה | איך לבדוק | ✓ |
|---|--------|-----------|---|
| P0 | Baseline ירוק | `npm run lint` + `npm test` + `npm run build` — ללא שגיאות | ☐ |
| P1 | Baseline חיישנים | `npm run dev` → מצלמה ON → `npm run qa:vision` | ☐ |
| P2 | תיעוד baseline | שמור `tests/qa-vision-premium-results.json` + צילום מסך Inspector | ☐ |
| P3 | Git branch | עבודה על branch ייעודי (`vision-2.0/phase-N`) | ☐ |

**פקודות מהירות (כל שלב):**

```bash
npm run lint
npm test
npm run build
npm run dev
# טרמינל נוסף:
npm run qa:vision
```

---

## רגרסיה חיישנים — חובה בכל שלב (Sensor UI Gate)

> **אם אחת מהבדיקות האלה נכשלת — השלב לא מאושר**, גם אם שכבות ההבנה עובדות.

| ID | בדיקה | צעדים | Pass |
|----|--------|-------|------|
| **S-GATE-1** | Vision Inspector נפתח | מצלמה ON → 🔬 Vision Inspector | כל 10 קבוצות כרטיסים נראות (Objects, Pose, Hands, Gestures, Body Language, Actions, Face, Emotion, Environment, Scene) | ☐ |
| **S-GATE-2** | FPS / pipeline | Inspector toolbar — FPS > 0 יציב ~10s | ☐ |
| **S-GATE-3** | YOLO objects | החזק חפץ (כוס / טלפון / laptop) | Objects card מתעדכן | ☐ |
| **S-GATE-4** | Hands + fingers | הרם יד, 1–5 אצבעות | Hands + Gestures + finger count מתעדכנים | ☐ |
| **S-GATE-5** | Pose | עמידה / ישיבה | Pose / Actions cards מתעדכנים | ☐ |
| **S-GATE-6** | Face + Emotion | פנים בפריים | Face card: age, gender, gaze; Emotion meter עם 6 bars | ☐ |
| **S-GATE-7** | Video overlay | Vision Inspector feed | bbox אובייקטים / פנים / ידיים (אם מופעל) | ☐ |
| **S-GATE-8** | Toggles | כבה/הדלק YOLO, Hands, Face ב-inspector | מודול מתכבה/נדלק; pipeline לא קורס | ☐ |
| **S-GATE-9** | Presets | balanced / lite / full | intervals משתנים; FPS לא קורס ל-0 | ☐ |
| **S-GATE-10** | Always-on | שלח הודעת צ'אט בזמן מצלמה | `pipelinePaused === false`; YOLO/objects ממשיכים (ראה `vision-premium-qa.md` A1–A5) | ☐ |
| **S-GATE-11** | World Memory sidebar | Inspector sidebar | objects, pose, gestures, fingerStates תואמים לכרטיסים | ☐ |
| **S-GATE-12** | אוטומציה | `npm run qa:vision` | `faceOk`, `emotionOk`, objects detected — כמו baseline | ☐ |

**קבצי UI שלא אמורים להישבר:** `CameraPreview.tsx`, `VisionInspectorPanel.tsx`, `VisionInspectorFeed.tsx`, `VisionDetectionOverlay.tsx`, `VisionDashboard.tsx`, `CameraVisionHud.tsx`.

---

## Phase 0 — חוזה + גדר (DialogueContext)

**מטרה:** הגדרת schema; Gemma לא מקבל raw sensors.

### יישום

| # | משימה | קובץ/אזור | ✓ |
|---|--------|-----------|---|
| 0.1 | הגדר `DialogueContext`, `WorldSnapshot`, `FrameBundle` (private) | `app/src/vision2/` (חדש) | ☐ |
| 0.2 | `serializeDialogueContext()` — JSON ל-LLM בלבד | `dialogueContext.ts` | ☐ |
| 0.3 | Feature flag `vision2.enabled` (settings / env) | `visionSettings.ts` / App | ☐ |
| 0.4 | כש-flag כבוי — התנהגות legacy ללא שינוי | App.tsx | ☐ |
| 0.5 | Unit tests ל-schema + serialization | `*.test.ts` | ☐ |

### QA Phase 0

| # | בדיקה | Pass | ✓ |
|---|--------|------|---|
| 0-QA-1 | `npm test` — tests חדשים ירוקים | ☐ |
| 0-QA-2 | Flag **כבוי** — צ'אט + מצלמה כמו לפני | ☐ |
| 0-QA-3 | **Sensor UI Gate** S-GATE-1…12 | ☐ |
| 0-QA-4 | Activity log: אין regression ב-YOLO lines | ☐ |

### Sign-off Phase 0

- [ ] Schema מוגדר; flag כבוי = zero regression
- [ ] Sensor UI Gate עבר

---

## Phase 1 — Perception Engine + Human State Engine

**מטרה:** חיישנים → עובדות (`touchingFace`, `raisedHand`); מצב אדם (posture, attention, activity, energy, engagement).

### יישום

| # | משימה | ✓ |
|---|--------|---|
| 1.1 | `PerceptionEngine` — wrap מ-`BodyLanguageInterpreter`, `InteractionAnalyzer`, `GestureRecognizer` | ☐ |
| 1.2 | Output typed `ObservationSet` — לא strings ל-WorldMemory ישירות | ☐ |
| 1.3 | `TemporalTracker` — duration ל-bool signals | ☐ |
| 1.4 | `HumanStateEngine` — EMA + `HumanState` | ☐ |
| 1.5 | `VisionPipeline` / `GroveeVisionRunner` — L1 נשאר; L2–L3 רצים אחרי `publishFrame` | ☐ |
| 1.6 | Inspector **ממשיך** לקבל `VisionResult` גolמי (רק פנימי) | ☐ |
| 1.7 | Unit tests: perception facts + state smoothing | ☐ |

### QA Phase 1

| # | בדיקה | צעדים | Pass | ✓ |
|---|--------|-------|------|---|
| 1-QA-1 | Perception unit | `npm test` perception + state | ☐ |
| 1-QA-2 | יד על פנים | 3s+ | `touchingFace: true` ב-dev probe / debug panel | ☐ |
| 1-QA-3 | ניפנוף | wave 2s | `raisedHand` / motion observation | ☐ |
| 1-QA-4 | כוס ביד | YOLO cup + hand | `holdingCup: true` | ☐ |
| 1-QA-5 | **Sensor UI Gate** | S-GATE-1…12 | ☐ |
| 1-QA-6 | `VisionResult` ל-Inspector | כרטיסים זהים ל-baseline | ☐ |
| 1-QA-7 | Flag off | legacy path | ☐ |

### Sign-off Phase 1

- [ ] Observations + HumanState ב-dev; Inspector לא נפגע
- [ ] Sensor UI Gate עבר

---

## Phase 2 — Body Language Model (וקטור הסתברותי)

**מטרה:** `{ focused, thinking, stressed, bored }` + confidence + ageSec — לא label קשיח.

### יישום

| # | משימה | ✓ |
|---|--------|---|
| 2.1 | `BodyLanguageModel` — weighted fusion | ☐ |
| 2.2 | Temporal decay + min dwell time | ☐ |
| 2.3 | כרטיס Body Language ב-Inspector — **אופציונלי:** הצג scores (לא חובה לשלב UI; לא לשבור כרטיס קיים) | ☐ |
| 2.4 | Unit tests: chin+stable gaze → thinking > 0.6; hands on head + motion → stressed | ☐ |

### QA Phase 2

| # | בדיקה | צעדים | Pass | ✓ |
|---|--------|-------|------|---|
| 2-QA-1 | Thinking pattern | יד על סנטר + מבט יציב 20s | thinking ≥ 0.55 | ☐ |
| 2-QA-2 | Stress pattern | שתי ידיים על ראש + תנועה | stressed עולה | ☐ |
| 2-QA-3 | Focus pattern | מסך + ישיבה יציבה | focused ≥ 0.6 | ☐ |
| 2-QA-4 | Bored pattern | מעט תנועה + היעדר engagement | bored עולה לאט | ☐ |
| 2-QA-5 | ageSec | מצב יציב 10s | ageSec ≥ 8 | ☐ |
| 2-QA-6 | **Sensor UI Gate** | S-GATE-1…12 | ☐ |
| 2-QA-7 | Body Language card (legacy cues) | עדיין מציג cues — לא ריק / לא crash | ☐ |

### Sign-off Phase 2

- [ ] Vector scores יציבים; לא flicker כל פריים
- [ ] Sensor UI Gate עבר

---

## Phase 3 — Situation Engine + World Model

**מטרה:** "Person working" / "Person drinking"; `WorldSnapshot` מאוחד.

### יישום

| # | משימה | ✓ |
|---|--------|---|
| 3.1 | `SituationEngine` — state machine + hysteresis | ☐ |
| 3.2 | Migrate `EventRuleEngine` + `situationRegistry` → situation.primary | ☐ |
| 3.3 | `WorldModel` — room + person + session | ☐ |
| 3.4 | `WorldMemory` — adapter read/write או deprecate הדרגתי | ☐ |
| 3.5 | Deep vision (`analyze_scene`) → `room.semanticNotes` בלבד, לא object dump לצ'אט | ☐ |
| 3.6 | World Memory panel — מציג World Model (או snapshot תואם) | ☐ |

### QA Phase 3

| # | בדיקה | Pass | ✓ |
|---|--------|------|---|
| 3-QA-1 | Laptop + person + ישיבה | situation.primary = working (conf ≥ 0.6) | ☐ |
| 3-QA-2 | Cup + hand | situation = drinking / break | ☐ |
| 3-QA-3 | Phone + face | situation = using_phone | ☐ |
| 3-QA-4 | Wave | situation = greeting (לא regression על speech) | ☐ |
| 3-QA-5 | Hysteresis | flicker object 1s | situation לא מתחלף כל שנייה | ☐ |
| 3-QA-6 | Boot deep | summary ב-bootContext; לא מוחק objects ב-Inspector | ☐ |
| 3-QA-7 | **Sensor UI Gate** | S-GATE-1…12 | ☐ |
| 3-QA-8 | Situation settings tab | toggle wave off → אין דיבור יזום wave | ☐ |

### Sign-off Phase 3

- [ ] WorldSnapshot עקבי; Inspector + sidebar תואמים
- [ ] Sensor UI Gate עבר

---

## Phase 4 — Coach Engine + Character Brain v2

**מטרה:** תגובה ל-meaning ו-coach intent; לא ל-COCO events.

### יישום

| # | משימה | ✓ |
|---|--------|---|
| 4.1 | `CoachEngine` — offerSupport, suggestBreak, encourage | ☐ |
| 4.2 | `CharacterBrain` v2 — input: WorldSnapshot + coach | ☐ |
| 4.3 | Proactive speech דרך `DialogueContext.character.shouldSpeak` | ☐ |
| 4.4 | `buildRichSensorBlock` → deprecated; `buildDialogueContext()` | ☐ |
| 4.5 | `resolveUtterance` — מקבל meaning, לא finger counts | ☐ |

### QA Phase 4

| # | בדיקה | צעדים | Pass | ✓ |
|---|--------|-------|------|---|
| 4-QA-1 | Stress coach | hands on head 15s+ stressed>0.75 | coach.intent = offer_support (או דיבור יזום מתאים) | ☐ |
| 4-QA-2 | Break coach | focused 45min+ (simulate / mock session) | suggest_break | ☐ |
| 4-QA-3 | Encourage | thumbs up + happy | encourage | ☐ |
| 4-QA-4 | Wave | wave | greeting — לא "I see arm" | ☐ |
| 4-QA-5 | holdForChat | שלח צ'אט | אין spam proactive; sensors רצים (A1) | ☐ |
| 4-QA-6 | **Sensor UI Gate** | S-GATE-1…12 | ☐ |
| 4-QA-7 | Activity log | character_speak עם reason situation:* / coach:* | ☐ |

### Sign-off Phase 4

- [ ] דיבור יזום מבוסס meaning; cooldowns עובדים
- [ ] Sensor UI Gate עבר

---

## Phase 5 — Dialogue Brain (חוזה LLM מלא)

**מטרה:** Gemma מקבל **רק** `DialogueContext` — אין landmarks / bbox / fingerStates ב-prompt.

### יישום

| # | משימה | ✓ |
|---|--------|---|
| 5.1 | `sendPrompt` — inject `serializeDialogueContext()` בלבד (flag on) | ☐ |
| 5.2 | הסר `buildRichSensorBlock` / `fingerStates` / raw cues מ-worker prompts | ☐ |
| 5.3 | `analyze_scene` — enrichment ל-WorldModel בלבד | ☐ |
| 5.4 | `FINGER_COUNT` — תשובה מ-Perception/state, לא dump landmarks ל-LLM | ☐ |
| 5.5 | Activity log: `camera_context` מציג JSON מסונן (ללא raw) | ☐ |
| 5.6 | E2E test / grep: אין `fingerStates` ב-prompt strings | ☐ |

### QA Phase 5 — צ'אט

| # | שאלה (עברית) | Expect | ✓ |
|---|--------------|--------|---|
| 5-QA-1 | `כמה אצבעות אתה רואה?` (2 אצבעות) | מספר נכון; לא "bbox" / לא רשימת חיישנים | ☐ |
| 5-QA-2 | `האדם עומד או יושב?` | תשובה tentative; FRESH state | ☐ |
| 5-QA-3 | `מה אתה רואה?` | פרשנות HAL; לא inventory שעון/כיסא/מיטה | ☐ |
| 5-QA-4 | `אתה רואה אותי?` | כן/לא לפי present | ☐ |
| 5-QA-5 | `היי` + מצלמה | ברכה קצרה + אווירה | ☐ |
| 5-QA-6 | Prompt audit | DevTools / Activity — **אין** landmarks, YOLO labels, fingerStates ב-system prompt | ☐ |

### QA Phase 5 — חיישנים

| # | בדיקה | ✓ |
|---|--------|---|
| 5-QA-7 | **Sensor UI Gate** S-GATE-1…12 | ☐ |
| 5-QA-8 | `npm run qa:vision` — זהה baseline ± tolerance | ☐ |

### Sign-off Phase 5

- [ ] LLM contract enforced
- [ ] Sensor UI Gate + qa:vision עבר

---

## Phase 6 — Memory + Episodic (הבנות, לא detections)

**מטרה:** `userFocusedFor: 12min`, episodes, לא `gesture: wave` גolמי.

### יישום

| # | משימה | ✓ |
|---|--------|---|
| 6.1 | `EpisodicMemory` — focus_block, stress_episode, break, greeting | ☐ |
| 6.2 | Durations + peak scores | ☐ |
| 6.3 | `recentChanges` ב-DialogueContext — semantic only | ☐ |
| 6.4 | Prune / TTL — לא גדל ללא הגבלה | ☐ |

### QA Phase 6

| # | בדיקה | Pass | ✓ |
|---|--------|------|---|
| 6-QA-1 | Focus 5min (mock/fast clock) | episodic focus_block.durationSec | ☐ |
| 6-QA-2 | Greeting | lastGreetingAt מתעדכן; לא spam | ☐ |
| 6-QA-3 | Person left | person layer cleared; episodes נשמרים | ☐ |
| 6-QA-4 | **Sensor UI Gate** | S-GATE-1…12 | ☐ |

### Sign-off Phase 6

- [ ] Memory מסכם הבנות; Inspector לא נפגע

---

## Phase 7 — יכולות עתידיות (תשתית)

**מטרה:** hooks ל-Audio, Social, Productivity Coach, Teaching, Emotional — בלי לשבור L1.

| # | יכולת | תשתית | QA מינימלי | ✓ |
|---|--------|--------|------------|---|
| 7.1 | Audio (L1) | `AudioSensor` stub + FrameBundle | מצלמה+חיישנים ללא regression | ☐ |
| 7.2 | Social Awareness | agreement / confusion scores | unit tests | ☐ |
| 7.3 | Productivity Coach | workSessionMin tracking | session counter ב-dev | ☐ |
| 7.4 | Teaching Assistant | attention_loss detection | unit test | ☐ |
| 7.5 | Emotional Companion | support intent thresholds | coach QA | ☐ |

**Sensor UI Gate חובה** אחרי כל יכולת חדשה: S-GATE-1…12.

---

## Acceptance סופי (Vision 2.0 Complete)

| # | קriterion | ✓ |
|---|----------|---|
| F1 | אין raw sensors ב-Gemma prompts (audit) | ☐ |
| F2 | `DialogueContext` הוא input יחיד לצ'אט | ☐ |
| F3 | יד על פנים 30s → thinking score, לא event "hand_on_face" | ☐ |
| F4 | Coach מציע break אחרי focus ארוך | ☐ |
| F5 | Inspector 10 cards + overlay + FPS — כמו baseline | ☐ |
| F6 | `npm test` + `npm run lint` + `npm run build` + `npm run qa:vision` | ☐ |
| F7 | `tests/manual-qa.md` V1–V8 + chat 1–8 | ☐ |
| F8 | Feature flag `vision2.enabled` default ON (אחרי אישור מוצר) | ☐ |

---

## Rollback

| שלב | פעולה |
|-----|--------|
| כל phase | `vision2.enabled = false` → legacy path |
| regression חמור | revert branch; baseline `qa-vision-premium-results.json` |
| Inspector broken | עדיפות 1: restore `VisionResult` path ל-UI; reasoning בשכבה נפרדת |

---

## יומן ביצוע (מלא ידנית)

| Phase | תאריך | מבצע | Sensor Gate | Tests | הערות |
|-------|--------|------|-------------|-------|--------|
| 0 | 2026-06-10 | agent | ☐ ידני | ☑ `vision2.test.ts` | schema + flag |
| 1 | 2026-06-10 | agent | ☐ ידני | ☑ | PerceptionEngine + HumanState |
| 2 | 2026-06-10 | agent | ☐ ידני | ☑ | BodyLanguageModel vector |
| 3 | 2026-06-10 | agent | ☐ ידני | ☑ | SituationEngine + WorldModel |
| 4 | 2026-06-10 | agent | ☐ ידני | ☑ | CoachEngine + runner hook |
| 5 | 2026-06-10 | agent | ☐ ידני | ☑ | App DialogueContext for chat |
| 6 | 2026-06-10 | agent | ☐ ידני | ☑ | EpisodicMemory |
| 7 | 2026-06-10 | agent | ☐ ידני | ☑ | Audio stub, Social, Productivity, Teaching, Emotional |
| **Final** | 2026-06-10 | agent | ☐ ידני | ☑ 121 tests | `npm run qa:vision` — ידני |

---

## קישורים

- QA חיישנים קיים: [vision-premium-qa.md](./vision-premium-qa.md)
- QA ידני כללי: [manual-qa.md](./manual-qa.md)
- Unit tests: `app/src/vision-always-on.test.ts`, `situationTriggerEngine.test.ts`, `visionBridge.test.ts`, `chatIntents.test.ts`
