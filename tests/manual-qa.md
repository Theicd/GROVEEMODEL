# Manual QA checklist (GROVEE)

Run after `npm run dev` or on the deployed site. Gemma 4 must finish downloading first.

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

**Note:** Automated tests (`npm test`) only cover intent routing helpers, not model quality.
