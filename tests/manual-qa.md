# Manual QA checklist (GROVEE)

Run after `npm run dev` or on the deployed site. Gemma 4 must finish downloading first.

## Chat & model

| # | Goal | Try (paste) | Expect |
|---|------|-------------|--------|
| 1 | Chat / Hebrew | `ספר משפט אחד על קפה.` | Short Hebrew reply, readable punctuation. |
| 2 | Greeting | `היי` | One short friendly line. |
| 3 | Story | `המשך את הסיפור: פעם אחת יצא דוב ליער.` | Coherent continuation in Hebrew. |
| 4 | Code | `כתוב פונקציית JavaScript שמחזירה סכום מערך.` | Gemma returns code (fenced block) in Hebrew context. |
| 5 | Search | Enable **Search**, ask `What is WebGPU?` in English. | Wikipedia snippets may inform the answer. |
| 6 | HTML preview | Ask for `דף HTML פשוט עם כותרת אדומה` | ` ```html ` block renders in iframe preview. |
| 7 | Settings | Open ⚙, change temperature, reload page | Values persist (`localStorage`). |
| 8 | Cache | Clear model cache, click **התחל** | Gemma re-downloads from Hugging Face. |

## Vision & HAL (camera mode)

| # | Goal | Steps | Expect |
|---|------|-------|--------|
| V1 | Vision Inspector cards | Camera on → 🔬 Vision Inspector | 10 card groups update (Objects, Pose, Hands, Gestures, Body Language, Actions, Face, Emotion, Environment, Scene). |
| V2 | Finger count | Hold 2 fingers → ask `כמה אצבעות אתה רואה?` | Answer references 2 fingers from sensor block. |
| V3 | World Memory panel | Vision Inspector sidebar | Objects, pose, gestures, finger counts match live cards. |
| V4 | Wave (no LLM polish) | Wave at camera | Hebrew proactive line within ~15s; Settings → proactive LLM polish **off** by default. |
| V5 | Situation settings | ⚙ → tab **עיניים ואסיטואציות** | Toggle rule off (e.g. wave) → wave no longer triggers speech. |
| V6 | Boot snapshot | Enable boot snapshot, restart camera | One Gemma deep pass at start; `World Memory` gets richer summary. |
| V7 | Activity log | Enable "רישום זיהוי ביומן" | Activity panel shows YOLO / vision lines while camera runs. |
| V8 | Performance preset | Settings → balanced/lite/full | Vision Inspector intervals change; FPS stable. |

**Note:** Automated tests (`npm test`) cover intent routing, situation triggers, and vision bridge helpers — not model quality.
