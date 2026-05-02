# Manual QA checklist (GROVEE)

Run after `npm run dev` or on the deployed site. Models must finish downloading first.

| # | Goal | Try (paste or attach) | Expect |
|---|------|------------------------|--------|
| 1 | Chat / Hebrew | `ספר משפט אחד על קפה.` | Short Hebrew reply, readable punctuation. |
| 2 | Greeting | `היי` | One short friendly line. |
| 3 | Story | `המשך את הסיפור: פעם אחת יצא דוב ליער.` | Coherent continuation in Hebrew. |
| 4 | Code path | `כתוב פונקציית JavaScript שמחזירה סכום מערך.` | Qwen output wrapped; Gemma summarizes in Hebrew + fenced code. |
| 5 | Code + Search | Enable **Search**, ask `What is WebGPU?` in English + ask for a tiny code sample. | Wikipedia snippets may appear in reasoning; code still present. |
| 6 | Image (cloud) | Settings → Image → Pollinations. `צור תמונה של ספינת חלל כחולה.` | English prompt internally; image + Hebrew explanation. |
| 7 | Image (local) | Settings → **Local SD-Turbo**, WebGPU browser. Same prompt. | First run ~2.3GB; 512×512 PNG; fallback message if WebGPU fails. |
| 8 | Vision | Attach a photo, `מה רואים בתמונה?` | Caption then Gemma polish in Hebrew. |
| 9 | HTML preview | Ask for `דף HTML פשוט עם כותרת אדומה` | ` ```html ` block renders in iframe preview. |
|10 | Settings | Open ⚙, change temperature, reload page | Values persist (`localStorage`). |

**Note:** Automated tests (`npm test`) only cover intent routing helpers, not model quality.
