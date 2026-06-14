# בדיקה חיה — תשובות ממשק

**תאריך:** 2026-06-14T00:13:58.055Z

| סטטוס | כמות |
|--------|------|
| ✅ תשובה מובנית מלאה | 78 |
| ⚠️ חלקי / LLM | 3 |
| ❌ נכשל | 2 |
| ⏭️ דילוג | 9 |

**סה"כ:** 92 שאלות

## רשימה מהירה

| # | ID | סטטוס | שאלה | נתיב | מקורות |
|---|-----|--------|------|------|--------|
| 1 | N01 | ✅ | מה הכותרות החשובות בעולם כרגע? | canned-live-reply | news-rss |
| 2 | N02 | ✅ | מה קורה בישראל היום? | canned-live-reply | news-rss |
| 3 | N03 | ✅ | האם יש אסונות טבע פעילים כרגע? | canned-live-reply | gdacs-disasters |
| 4 | N04 | ✅ | אילו מדינות נמצאות תחת התרעות מזג אוויר? | globe-focusIsrael | — |
| 5 | N05 | ✅ | האם התרחשו רעידות אדמה משמעותיות ב-24 השעות ה | canned-live-reply | usgs-earthquake |
| 6 | N06 | ✅ | מהי השריפה הגדולה ביותר הפעילה כרגע בעולם? | canned-live-reply | gdacs-disasters |
| 7 | N07 | ✅ | אילו סופות טרופיות פעילות כרגע? | canned-live-reply | gdacs-disasters |
| 8 | E01 | ✅ | הצג את 20 רעידות האדמה האחרונות בעולם | canned-live-reply | usgs-earthquake |
| 9 | E02 | ✅ | מה הייתה רעידת האדמה החזקה ביותר השבוע? | canned-live-reply | usgs-earthquake |
| 10 | E03 | ✅ | הצג אותן על הגלובוס | globe-place (canned) | — |
| 11 | E04 | ✅ | האם הייתה רעידת אדמה ליד יפן ב-48 השעות האחרו | canned-live-reply | usgs-earthquake |
| 12 | E05 | ⏭️ | כמה רעידות מעל 5.0 התרחשו החודש? | no-search | — |
| 13 | W01 | ✅ | מה גובה הגלים מול חופי תל אביב? | canned-live-reply | open-meteo-marine |
| 14 | W02 | ⚠️ | איפה נמצאים הגלים הגבוהים ביותר כרגע בעולם? | globe-showLayer | — |
| 15 | W03 | ✅ | מה הטמפרטורה באיסלנד עכשיו? | canned-live-reply | open-meteo |
| 16 | W04 | ⏭️ | איזה אזור בעולם חווה את הרוחות החזקות ביותר כ | no-search | — |
| 17 | W05 | ✅ | הצג מפה של מזג האוויר באירופה | canned-live-reply | open-meteo |
| 18 | A01 | ✅ | כמה מטוסים נמצאים כרגע מעל ישראל? | canned-live-reply | adsb-aviation |
| 19 | A02 | ✅ | כמה מטוסים מעל לונדון? | canned-live-reply | adsb-aviation |
| 20 | A03 | ⏭️ | אילו טיסות מתקרבות לנחיתה בנתב"ג? | no-search | — |
| 21 | A04 | ⏭️ | מהו שדה התעופה העמוס ביותר כרגע? | no-search | — |
| 22 | A05 | ✅ | הצג את כל המטוסים מעל הים התיכון | canned-live-reply | adsb-aviation |
| 23 | M01 | ✅ | כמה מטוסים צבאיים מזוהים כרגע באזור הים התיכו | canned-live-reply | adsb-aviation |
| 24 | M02 | ✅ | האם יש מטוסי תדלוק אווירי מעל אירופה? | canned-live-reply | adsb-aviation |
| 25 | M03 | ✅ | אילו מטוסי AWACS פעילים כרגע? | canned-live-reply | adsb-aviation |
| 26 | M04 | ✅ | הצג את מיקומם על המפה | globe-place (canned) | — |
| 27 | S01 | ✅ | כמה אוניות נמצאות במפרץ סואץ? | canned-live-reply | ais-ships |
| 28 | S02 | ✅ | כמה מכליות נפט נמצאות במפרץ הפרסי? | canned-live-reply | ais-ships |
| 29 | S03 | ✅ | אילו אוניות נמצאות ליד חופי ישראל? | canned-live-reply | ais-ships |
| 30 | S04 | ✅ | הצג אוניות מכולה באזור רוטרדם | canned-live-reply | ais-ships |
| 31 | S05 | ⏭️ | מהו הנמל העמוס ביותר כרגע? | no-search | — |
| 32 | S06 | ✅ | כמה כלי שייט או אוניות יש במפרץ חיפה? | canned-live-reply | ais-ships |
| 33 | S07 | ❌ | כמה מצופים יש במפרץ חיפה? | web-search | — |
| 34 | SP01 | ✅ | איפה נמצאת תחנת החלל כרגע? | canned-live-reply | iss-tracker |
| 35 | SP02 | ✅ | מתי היא תעבור מעל ישראל? | canned-live-reply | iss-tracker |
| 36 | SP03 | ⏭️ | אילו לווייני Starlink נמצאים מעל אירופה? | no-search | — |
| 37 | SP04 | ❌ | כמה לוויינים פעילים במסלול נמוך? | web-search | — |
| 38 | SP05 | ✅ | הצג את מסלול ה-ISS על הגלובוס | canned-live-reply | iss-tracker |
| 39 | G01 | ✅ | איפה נמצאת גרמניה? | globe-focusPlaceQuiet | — |
| 40 | G02 | ✅ | התקרב לברלין | globe-focusPlaceQuiet | — |
| 41 | G03 | ✅ | הצג את פריז | globe-place (canned) | — |
| 42 | G04 | ✅ | הצג את הר האוורסט | globe-place (canned) | — |
| 43 | G05 | ✅ | הצג את תעלת פנמה | globe-place (canned) | — |
| 44 | G06 | ✅ | הצג את משולש ברמודה | globe-place (canned) | — |
| 45 | T01 | ⏭️ | מהי תחנת הרכבת הקרובה ביותר לשדה התעופה בברלי | no-search | — |
| 46 | T02 | ✅ | הצג אותה על המפה | globe-place (canned) | — |
| 47 | T03 | ⏭️ | כמה זמן נסיעה משם למרכז העיר? | no-search | — |
| 48 | T04 | ⏭️ | אילו קווי רכבת מגיעים לשם? | no-search | — |
| 49 | GH01 | ✅ | מצא פרויקטי WebGPU חדשים | canned-live-reply | github |
| 50 | GH02 | ✅ | מצא פרויקטי AI שפורסמו השבוע | canned-live-reply | github |
| 51 | GH03 | ✅ | מהם הפרויקטים הפופולריים ביותר היום? | canned-live-reply | github, hacker-news |
| 52 | GH04 | ⚠️ | מצא משחקים שנבנו עם Three.js | game-panel + archive.org | — |
| 53 | GH05 | ✅ | מצא חלופות ל-Ollama | canned-live-reply | github |
| 54 | HF01 | ✅ | מצא מודלים חדשים השבוע | canned-live-reply | github, huggingface-models, huggingface- |
| 55 | HF02 | ✅ | מהם מודלי ה-VLM הפופולריים ביותר? | canned-live-reply | huggingface-models, huggingface-datasets |
| 56 | HF03 | ✅ | מצא מודלים לזיהוי אובייקטים | canned-live-reply | github, huggingface-models, huggingface- |
| 57 | HF04 | ✅ | מצא מודלים לזיהוי תנוחות גוף | canned-live-reply | github, huggingface-models, huggingface- |
| 58 | HF05 | ✅ | מצא מודלים ל-WebGPU | canned-live-reply | github, huggingface-models, huggingface- |
| 59 | HF06 | ✅ | מצא מודלים שמתאימים להרצה בדפדפן | canned-live-reply | github |
| 60 | GM01 | ✅ | שחק Doom | game-panel + archive.org | archive.org |
| 61 | GM02 | ✅ | שחק Doom II | game-panel + archive.org | archive.org |
| 62 | GM03 | ✅ | שחק Dune II | game-panel + archive.org | archive.org |
| 63 | GM04 | ✅ | שחק Prince of Persia | game-panel + archive.org | archive.org |
| 64 | GM05 | ✅ | שחק Wolfenstein 3D | game-panel + archive.org | archive.org |
| 65 | GM06 | ✅ | מצא משחקי DOS אסטרטגיה | game-panel + archive.org | archive.org |
| 66 | GM07 | ⚠️ | מצא משחקי SEGA משנות ה-90 | game-panel + archive.org | — |
| 67 | F01 | ✅ | כמה מטוסים נמצאים מעל האזור שבו נמצאת כרגע הס | canned-live-reply | adsb-aviation |
| 68 | F02 | ✅ | האם יש אוניות באזורי התרעת צונאמי? | canned-live-reply | ais-ships, gdacs-disasters |
| 69 | F03 | ✅ | האם תחנת החלל נמצאת כרגע מעל מדינה שבה יש סער | canned-live-reply | iss-tracker |
| 70 | F04 | ✅ | הצג את כל רעידות האדמה שהתרחשו בטווח של 500 ק | canned-live-reply | usgs-earthquake |
| 71 | F05 | ✅ | אילו שדות תעופה נמצאים במסלול של סופת הוריקן  | canned-live-reply | gdacs-disasters |
| 72 | O01 | ✅ | מה הדברים המעניינים שקורים בעולם עכשיו? | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 73 | O02 | ✅ | האם יש משהו חריג שמתרחש כרגע? | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 74 | O03 | ✅ | אילו אירועים חשובים התרחשו ב-24 השעות האחרונו | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 75 | O04 | ✅ | תן לי סקירה של מצב העולם כרגע | canned-live-reply | open-meteo, open-meteo-marine, usgs-eart |
| 76 | O05 | ✅ | הצג לי את המקומות הפעילים ביותר על פני כדור ה | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 77 | O06 | ✅ | מה קורה עכשיו בחלל? | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 78 | O07 | ✅ | מה קורה עכשיו באוקיינוסים? | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 79 | O08 | ✅ | מה קורה עכשיו בשמי אירופה? | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 80 | ST01 | ✅ | תן לי תמונת מצב מלאה של כדור הארץ כרגע | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 81 | ST02 | ✅ | מה 20 האירועים החריגים ביותר שמתרחשים עכשיו? | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 82 | ST03 | ✅ | הצג על הגלובוס בו זמנית מטוסים, אוניות, רעידו | canned-live-reply | usgs-earthquake, ais-ships, adsb-aviatio |
| 83 | ST04 | ✅ | סכם את כל ההתראות הפעילות בעולם | canned-live-reply | usgs-earthquake, ais-ships, news-rss, ad |
| 84 | L01 | ✅ | הצג רעידות אדמה על הגלובוס | canned-live-reply | usgs-earthquake |
| 85 | L02 | ✅ | מה מהירות הרוח בפריז? | canned-live-reply | open-meteo |
| 86 | L03 | ✅ | מתי תחנת החלל תעבור מעל ישראל? | canned-live-reply | iss-tracker |
| 87 | L04 | ✅ | הצג על המפה את גרמניה | globe-place (canned) | — |
| 88 | L05 | ✅ | כמה ק"מ בין ירושלים לחיפה? | globe-focusPlaceQuiet | — |
| 89 | L06 | ✅ | חפש פרויקטים בנושא WebGPU | canned-live-reply | github, wikipedia-he |
| 90 | L07 | ✅ | כמה שקלים שווים 100 דולר? | canned-live-reply | frankfurter-fx |
| 91 | L08 | ✅ | כמה תושבים יש בקנדה? | globe-focusPlaceQuiet | — |
| 92 | L09 | ✅ | מה מחיר הביטקוין עכשיו? | canned-live-reply | coingecko |

## פירוט מלא

### 1. N01 — ✅ מה הכותרות החשובות בעולם כרגע?

- **נתיב:** canned-live-reply
- **Intents:** news
- **מקורות:** news-rss
- **זמן:** 349ms

```
לפי עדכוני RSS:
• מקור: BBC News
• כותרות עדכניות (2026-06-14T00:10 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
3. Molly Russell's dad says PM rushing social media restrictions 'deplorable'
4. The nuclear challenge at the heart of Trump's Iran negotiations
5. Ticketmaster says Knicks fans won't be locked out of game after last-minute panic
מקור: חדשות (RSS).
```

### 2. N02 — ✅ מה קורה בישראל היום?

- **נתיב:** canned-live-reply
- **Intents:** news
- **מקורות:** news-rss
- **זמן:** 709ms

```
לפי עדכוני RSS:
• מקור: ynet
• כותרות עדכניות (2026-06-14T00:10 UTC):
1. תיעוד מחריד מברזיל: השליכו את הצעירה מהגשר - וגילו שלא קשרו את חבל הבאנג'י
2. מרוץ לחתימה, לפני ש"משהו ישתבש": זה מה שידוע על ההסכם עם איראן - ומה שלא
3. הלך לעולמו בגיל 40 אלוף-משנה נתנאל לסרי, לאחר מאבק במחלת הסרטן
4. בין כניעה לבגידה
5. מצפון תיפתח האחדות
מקור: חדשות (RSS).
```

### 3. N03 — ✅ האם יש אסונות טבע פעילים כרגע?

- **נתיב:** canned-live-reply
- **Intents:** disaster
- **מקורות:** gdacs-disasters
- **זמן:** 2181ms

```
לפי GDACS (אסונות טבע):
• אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green
4.  · Iceland · Green
5.  · Mexico · Green
6.  · Italy · Green
7.  · Philippines · Green
8.  · Indonesia · Green
מקור: אסונות (GDACS).
```

### 4. N04 — ✅ אילו מדינות נמצאות תחת התרעות מזג אוויר?

- **נתיב:** globe-focusIsrael
- **Intents:** weather, alerts, country
- **מקורות:** —
- **זמן:** 1ms

```
פאנל גלובוס: focusIsrael
```

### 5. N05 — ✅ האם התרחשו רעידות אדמה משמעותיות ב-24 השעות האחרונות?

- **נתיב:** canned-live-reply
- **Intents:** earthquake
- **מקורות:** usgs-earthquake
- **זמן:** 63ms

```
לפי USGS (רעידות אדמה):
• אין רעידות אדמה מדווחות ב-24 שעות באזור התרחשו, משמעותיות, ב-24 (USGS).
• לא נמצאו רעידות באזור (התרחשו, משמעותיות, ב-24) ב-24 שעות.
מקור: רעידות אדמה (USGS).
```

### 6. N06 — ✅ מהי השריפה הגדולה ביותר הפעילה כרגע בעולם?

- **נתיב:** canned-live-reply
- **Intents:** disaster
- **מקורות:** gdacs-disasters
- **זמן:** 467ms

```
לפי GDACS (אסונות טבע):
• אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green
4.  · Iceland · Green
5.  · Mexico · Green
6.  · Italy · Green
7.  · Philippines · Green
8.  · Indonesia · Green
מקור: אסונות (GDACS).
```

### 7. N07 — ✅ אילו סופות טרופיות פעילות כרגע?

- **נתיב:** canned-live-reply
- **Intents:** disaster
- **מקורות:** gdacs-disasters
- **זמן:** 441ms

```
לפי GDACS (אסונות טבע):
• אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green
4.  · Iceland · Green
5.  · Mexico · Green
6.  · Italy · Green
7.  · Philippines · Green
8.  · Indonesia · Green
מקור: אסונות (GDACS).
```

### 8. E01 — ✅ הצג את 20 רעידות האדמה האחרונות בעולם

- **נתיב:** canned-live-reply
- **Intents:** earthquake
- **מקורות:** usgs-earthquake
- **זמן:** 15ms

```
לפי USGS (רעידות אדמה):
• סה"כ 198 רעידות ב-24 שעות (USGS). 8 הגדולות:
- M5.2 · 2 km WSW of Kablalan, Philippines · 2026-06-13 02:05:53 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssls
- M5.1 · 2 km S of Cerro de Piedra, Mexico · 2026-06-13 18:20:42 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssq9
- M5.1 · 118 km ESE of Bitung, Indonesia · 2026-06-13 12:05:34 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssp2
- M5.1 · 207 km WSW of Adak, Alaska · 2026-06-13 11:24:54 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssnz
- M5.1 · 0 km ESE of Suyo, Peru · 2026-06-13 06:35:39 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssms
- M5.0 · 33 km ESE of Sarangani, Philippines · 2026-06-13 20:51:55 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssqy
- M5.0 · 24 km W of Ovalle, Chile · 2026-06-13 12:04:14 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssp1
- M5.0 · 93 km SSW of Unalaska, Alaska · 2026-06-13 10:38:47 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssnv
מקור: רעידות אדמה (USGS).
```

### 9. E02 — ✅ מה הייתה רעידת האדמה החזקה ביותר השבוע?

- **נתיב:** canned-live-reply
- **Intents:** earthquake
- **מקורות:** usgs-earthquake
- **זמן:** 63ms

```
לפי USGS (רעידות אדמה):
• סה"כ 2239 רעידות ב-7 ימים (USGS). 8 הגדולות:
• הרעידה החזקה ביותר: M7.8 · 26 km SW of Kablalan, Philippines
- M7.8 · 26 km SW of Kablalan, Philippines · 2026-06-07 23:37:41 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000srb1
- M6.5 · 20 km WSW of Balangonan, Philippines · 2026-06-08 00:55:11 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000srcg
- M6.1 · 102 km WNW of Mantua, Cuba · 2026-06-08 18:00:27 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000srjx
- M6.1 · 124 km SE of Severo-Kuril’sk, Russia · 2026-06-07 10:41:58 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000sr7l
- M6.0 · Auckland Islands, New Zealand region · 2026-06-10 00:44:21 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000srxn
- M6.0 · 6 km SSW of Pangyan, Philippines · 2026-06-07 23:49:14 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000src3
- M6.0 · 16 km WSW of Balangonan, Philippines · 2026-06-07 23:48:11 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000srbe
- M5.8 · 25 km ESE of Sarangani, Philippines · 2026-06-08 00:58:19 UTC
•   https://earthquake.usgs.gov/earthquakes/eventpage/us7000srci
מקור: רעידות אדמה (USGS).
```

### 10. E03 — ✅ הצג אותן על הגלובוס

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 1ms

```
הצגתי לך את ותן על הגלובוס על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 11. E04 — ✅ האם הייתה רעידת אדמה ליד יפן ב-48 השעות האחרונות?

- **נתיב:** canned-live-reply
- **Intents:** earthquake
- **מקורות:** usgs-earthquake
- **זמן:** 18ms

```
לפי USGS (רעידות אדמה):
• אין רעידות אדמה מדווחות ב-24 שעות באזור japan, honshu, hokkaido (USGS).
• לא נמצאו רעידות באזור (japan, honshu, hokkaido) ב-24 שעות.
מקור: רעידות אדמה (USGS).
```

### 12. E05 — ⏭️ כמה רעידות מעל 5.0 התרחשו החודש?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 13. W01 — ✅ מה גובה הגלים מול חופי תל אביב?

- **נתיב:** canned-live-reply
- **Intents:** marine
- **מקורות:** open-meteo-marine
- **זמן:** 274ms

```
לפי Open-Meteo Marine (גלים):
• מיקום: Tel Aviv, Tel Aviv, IL
• זמן: 2026-06-14T03:00
• גובה גל: 0.52 m
• כיוון גל: 301°
• מחזור גל: 5.05 s
• גובה swell: 0.52 m
• זרם ים: 0.4 km/h
מקור: ים וגלים (Open-Meteo Marine).
```

### 14. W02 — ⚠️ איפה נמצאים הגלים הגבוהים ביותר כרגע בעולם?

- **נתיב:** globe-showLayer
- **Intents:** marine
- **מקורות:** —
- **זמן:** 351ms

```
פתחתי את פאנל עולם חי (REALITY LIVE) — פקודה: showLayer · שכבה: marine.
```

### 15. W03 — ✅ מה הטמפרטורה באיסלנד עכשיו?

- **נתיב:** canned-live-reply
- **Intents:** weather
- **מקורות:** open-meteo
- **זמן:** 4074ms

```
לפי Open-Meteo (מזג אוויר):
• מיקום: איסלנד, IS
• גובה: 859 m
• זמן (מקומי): 2026-06-14T00:00
• מצב: מעונן
• טמפרatura: 6.7°C (מרגיש 4.4°C)
• לחות: 79%
• רוח: 5 km/h, כיוון 186°
• לחץ: 925.8 hPa
• תחזית 3 ימים:
- 2026-06-14 (2026-06-14): 2.4–13.3°C, גשם 0 mm (סיכוי 45%) · מעונן, רוח עד 23.8 km/h
- 2026-06-15 (2026-06-15): 1.8–11.9°C, גשם 0 mm (סיכוי 20%) · מעונן, רוח עד 31 km/h
- 2026-06-16 (2026-06-16): -0.2–7.5°C, גשם 0 mm (סיכוי 4%) · מעונן, רוח עד 31.6 km/h
מקור: מזג אוויר (Open-Meteo).
```

### 16. W04 — ⏭️ איזה אזור בעולם חווה את הרוחות החזקות ביותר כרגע?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 1ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 17. W05 — ✅ הצג מפה של מזג האוויר באירופה

- **נתיב:** canned-live-reply
- **Intents:** weather
- **מקורות:** open-meteo
- **זמן:** 150ms

```
לפי Open-Meteo (מזג אוויר):
• מיקום: אירופה
• גובה: 443 m
• זמן (מקומי): 2026-06-14T02:00
• מצב: שמיים בהירים
• טמפרatura: 16.3°C (מרגיש 15.4°C)
• לחות: 68%
• רוח: 6.5 km/h, כיוון 264°
• לחץ: 965.9 hPa
• תחזית 3 ימים:
- 2026-06-14 (2026-06-14): 11.9–21.5°C, גשם 0 mm (סיכוי 0%) · מעונן, רוח עד 20.8 km/h
- 2026-06-15 (2026-06-15): 10.8–20.2°C, גשם 0 mm (סיכוי 3%) · מעונן, רוח עד 11.5 km/h
- 2026-06-16 (2026-06-16): 10.3–25.4°C, גשם 0 mm (סיכוי 15%) · מעונן, רוח עד 12.2 km/h
מקור: מזג אוויר (Open-Meteo).
```

### 18. A01 — ✅ כמה מטוסים נמצאים כרגע מעל ישראל?

- **נתיב:** canned-live-reply
- **Intents:** aviation
- **מקורות:** adsb-aviation
- **זמן:** 257ms

```
לפי ADS-B חי:
• אזור: ישראל (מרכז) · רדיוס 250km
• מטוסים בטווח: 24
1. THY646 · גובה 35000ft · 457kn
2. FHY7087 · גובה 34000ft · 429kn
3. THY21Y · גובה 37000ft · 466kn
4. ROT101C · גובה 29075ft · 474kn
5. THY096 · גובה 33025ft · 443kn
6. THY81K · גובה 40000ft · 535kn
7. THY660 · גובה 36000ft · 464kn
8. THY577 · גובה 38000ft · 431kn
9. ADY402 · גובה 32275ft · 517kn
10. ELY844 · גובה 37000ft · 500kn
11. LOT151 · גובה 35000ft · 487kn
12. UAE926 · גובה 21900ft · 477kn
מקור: תעופה (ADS-B).
```

### 19. A02 — ✅ כמה מטוסים מעל לונדון?

- **נתיב:** canned-live-reply
- **Intents:** aviation
- **מקורות:** adsb-aviation
- **זמן:** 204ms

```
לפי ADS-B חי:
• אזור: לונדון · רדיוס 250km
• מטוסים בטווח: 156
1. RYR5EE · גובה 16075ft · 300kn
2. EXS21U · גובה 36000ft · 420kn
3. EIN58R · גובה 9625ft · 319kn
4. EIN4RL · גובה groundft · 48kn
5. EIN427 · גובה groundft · 23kn
6. RYR9YH · גובה groundft · 9kn
7. EIN799 · גובה 1975ft · 204kn
8. EIN459 · גובה 5275ft · 240kn
9. RYR4XH · גובה 14975ft · 291kn
10. RYR6YQ · גובה 9675ft · 262kn
11. EXS15CF · גובה 36000ft · 433kn
12. RYR4270 · גובה 13550ft · 266kn
מקור: תעופה (ADS-B).
```

### 20. A03 — ⏭️ אילו טיסות מתקרבות לנחיתה בנתב"ג?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 1ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 21. A04 — ⏭️ מהו שדה התעופה העמוס ביותר כרגע?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 22. A05 — ✅ הצג את כל המטוסים מעל הים התיכון

- **נתיב:** canned-live-reply
- **Intents:** aviation
- **מקורות:** adsb-aviation
- **זמן:** 279ms

```
לפי ADS-B חי:
• אזור: ברירת מחדל (NYC) · רדיוס 250km
• מטוסים בטווח: 639
1. N445ME · גובה 2150ft · 119kn
2. N512TR · גובה 1400ft · 96kn
3. PDT5862 · גובה 24975ft · 438kn
4. GJS4432 · גובה 19775ft · 372kn
5. N855ME · גובה 2775ft · 142kn
6. SWA2621 · גובה 28975ft · 514kn
7. AAL2534 · גובה 35975ft · 531kn
8. UAL2131 · גובה 18475ft · 407kn
9. AAL2689 · גובה 26000ft · 402kn
10. UAL1084 · גובה 33000ft · 541kn
11. N600AV · גובה 3025ft · 111kn
12. AAL1848 · גובה 25000ft · 409kn
מקור: תעופה (ADS-B).
```

### 23. M01 — ✅ כמה מטוסים צבאיים מזוהים כרגע באזור הים התיכון?

- **נתיב:** canned-live-reply
- **Intents:** aviation
- **מקורות:** adsb-aviation
- **זמן:** 98ms

```
לפי ADS-B חי:
• אזור: ברירת מחדל (NYC) · רדיוס 250km
• מטוסים בטווח: 639
1. N445ME · גובה 2150ft · 118kn
2. N512TR · גובה 1400ft · 96kn
3. PDT5862 · גובה 24975ft · 438kn
4. GJS4432 · גובה 19800ft · 372kn
5. N855ME · גובה 2775ft · 142kn
6. AAL2534 · גובה 35975ft · 530kn
7. SWA2621 · גובה 28975ft · 514kn
8. UAL2131 · גובה 18450ft · 406kn
9. AAL2689 · גובה 26000ft · 402kn
10. UAL1084 · גובה 33000ft · 541kn
11. N600AV · גובה 3025ft · 111kn
12. UWD83 · גובה 14675ft · 388kn
• הערה: ADS-B ציבורי לא מסמן באופן אמין מטוסים צבאיים; אין ספירה מדויקת של צבאי/אזרחי מהמקור הזה.
מקור: תעופה (ADS-B).
```

### 24. M02 — ✅ האם יש מטוסי תדלוק אווירי מעל אירופה?

- **נתיב:** canned-live-reply
- **Intents:** aviation
- **מקורות:** adsb-aviation
- **זמן:** 81ms

```
לפי ADS-B חי:
• אזור: מרכז אירופה · רדיוס 250km
• מטוסים בטווח: 106
1. EXS31ER · גובה 36000ft · 391kn
2. TOM12Y · גובה 38000ft · 366kn
3. EXS21LK · גובה 36000ft · 359kn
4. BEL6AK · גובה groundft · 0kn
5. TOM213 · גובה 38000ft · 369kn
6. OH68 · גובה groundft · 12kn
7. OH42 · גובה groundft · —
8. MPH9452 · גובה groundft · 5kn
9. P2 · גובה groundft · —
10. @@@@@@@@ · גובה groundft · 1kn
11. TRA40L · גובה groundft · 0kn
12. C2 · גובה groundft · 2kn
מקור: תעופה (ADS-B).
```

### 25. M03 — ✅ אילו מטוסי AWACS פעילים כרגע?

- **נתיב:** canned-live-reply
- **Intents:** aviation
- **מקורות:** adsb-aviation
- **זמן:** 96ms

```
לפי ADS-B חי:
• אזור: ברירת מחדל (NYC) · רדיוס 250km
• מטוסים בטווח: 639
1. N445ME · גובה 2150ft · 118kn
2. N512TR · גובה 1400ft · 96kn
3. PDT5862 · גובה 24975ft · 438kn
4. GJS4432 · גובה 19800ft · 372kn
5. N855ME · גובה 2775ft · 142kn
6. AAL2534 · גובה 35975ft · 530kn
7. SWA2621 · גובה 28975ft · 514kn
8. UAL2131 · גובה 18450ft · 406kn
9. AAL2689 · גובה 26000ft · 402kn
10. UAL1084 · גובה 33000ft · 541kn
11. N600AV · גובה 3025ft · 111kn
12. UWD83 · גובה 14675ft · 388kn
מקור: תעופה (ADS-B).
```

### 26. M04 — ✅ הצג את מיקומם על המפה

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 1ms

```
הצגתי לך את מיקומם על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 27. S01 — ✅ כמה אוניות נמצאות במפרץ סואץ?

- **נתיב:** canned-live-reply
- **Intents:** ships
- **מקורות:** ais-ships
- **זמן:** 971ms

```
לפי נתוני עולם חי / Digitraffic לגבי תעלת סואץ:
ספינות בטווח: 2 (0 AIS חי + 2 סימוני מסלול + 0 עולם חי)
דוגמאות:
• Suez Transit · מסלול · 31.25,32.31 · — → EGPSD
• Suez South · מסלול · 30.00,32.58 · — → EGPSD
מקור: עולם חי (AIS / סימוני מסלול).
```

### 28. S02 — ✅ כמה מכליות נפט נמצאות במפרץ הפרסי?

- **נתיב:** canned-live-reply
- **Intents:** ships
- **מקורות:** ais-ships
- **זמן:** 528ms

```
לפי נתוני עולם חי / Digitraffic לגבי מפרץ הפרסי:
ספינות בטווח: 2 (0 AIS חי + 2 סימוני מסלול + 0 עולם חי)
דוגמאות:
• Gulf Carrier · מסלול · 26.50,52.00 · — → AEJEA
• Dubai Jebel Ali · מסלול · 25.01,55.06 · — → AEJEA
מקור: עולם חי (AIS / סימוני מסלול).
```

### 29. S03 — ✅ אילו אוניות נמצאות ליד חופי ישראל?

- **נתיב:** canned-live-reply
- **Intents:** ships, places
- **מקורות:** ais-ships
- **זמן:** 387ms

```
לפי נתוני עולם חי / Digitraffic סביב ישראל (חוף):
ספינות בטווח: 5 (0 AIS חי + 5 סימוני מסלול + 0 עולם חי)
דוגמאות:
• Haifa Cargo · מסלול · 32.82,35.00 · — → ILHFA
• Haifa Port Route · מסלול · 32.79,35.02 · — → ILHFA
• Ashdod Container · מסלול · 31.83,34.63 · — → ILASH
• Eilat Tanker · מסלול · 29.55,34.96 · — → ILEIL
מקור: עולם חי (AIS / סימוני מסלול).
```

### 30. S04 — ✅ הצג אוניות מכולה באזור רוטרדם

- **נתיב:** canned-live-reply
- **Intents:** ships
- **מקורות:** ais-ships
- **זמן:** 530ms

```
לפי נתוני עולם חי / Digitraffic סביב נמל רוטרדם:
ספינות בטווח: 1 (0 AIS חי + 1 סימוני מסלול + 0 עולם חי)
דוגמאות:
• Rotterdam · מסלול · 51.92,4.48 · — → NLRTM
מקור: עולם חי (AIS / סימוני מסלול).
```

### 31. S05 — ⏭️ מהו הנמל העמוס ביותר כרגע?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 32. S06 — ✅ כמה כלי שייט או אוניות יש במפרץ חיפה?

- **נתיב:** canned-live-reply
- **Intents:** ships
- **מקורות:** ais-ships
- **זמן:** 307ms

```
לפי נתוני עולם חי / Digitraffic לגבי מפרץ חיפה:
ספינות בטווח: 2 (0 AIS חי + 2 סימוני מסלול + 0 עולם חי)
דוגמאות:
• Haifa Cargo · מסלול · 32.82,35.00 · — → ILHFA
• Haifa Port Route · מסלול · 32.79,35.02 · — → ILHFA
מקור: עולם חי (AIS / סימוני מסלול).
```

### 33. S07 — ❌ כמה מצופים יש במפרץ חיפה?

- **נתיב:** web-search
- **Intents:** marine-infra
- **מקורות:** —
- **זמן:** 1946ms

**שגיאה:** HTTP 429 for https://overpass.kumi.systems/api/interpreter

```
(ריק)
```

### 34. SP01 — ✅ איפה נמצאת תחנת החלל כרגע?

- **נתיב:** canned-live-reply
- **Intents:** satellite
- **מקורות:** iss-tracker
- **זמן:** 7593ms

```
לפי מעקב ISS:
• מיקום ISS (זמן אמת):
• קו רוחב: -43.53°
• קו אורך: -80.75°
• גובה: 432 km
• מהירות: 27552 km/h
מקור: תחנת חלל (ISS / עולם חי).
```

### 35. SP02 — ✅ מתי היא תעבור מעל ישראל?

- **נתיב:** canned-live-reply
- **Intents:** satellite
- **מקורות:** iss-tracker
- **זמן:** 6113ms

```
לפי מעקב ISS:
• מיקום ISS (זמן אמת):
• קו רוחב: -43.73°
• קו אורך: -80.32°
• גובה: 432 km
• מהירות: 27552 km/h
מקור: תחנת חלל (ISS / עולם חי).
```

### 36. SP03 — ⏭️ אילו לווייני Starlink נמצאים מעל אירופה?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 37. SP04 — ❌ כמה לוויינים פעילים במסלול נמוך?

- **נתיב:** web-search
- **Intents:** satellite
- **מקורות:** —
- **זמן:** 10006ms

**שגיאה:** fetch failed

```
(ריק)
```

### 38. SP05 — ✅ הצג את מסלול ה-ISS על הגלובוס

- **נתיב:** canned-live-reply
- **Intents:** satellite
- **מקורות:** iss-tracker
- **זמן:** 7241ms

```
לפי מעקב ISS:
• מיקום ISS (זמן אמת):
• קו רוחב: -44.32°
• קו אורך: -79.01°
• גובה: 432 km
• מהירות: 27551 km/h
מקור: תחנת חלל (ISS / עולם חי).
```

### 39. G01 — ✅ איפה נמצאת גרמניה?

- **נתיב:** globe-focusPlaceQuiet
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
פאנל גלובוס: focusPlaceQuiet
```

### 40. G02 — ✅ התקרב לברלין

- **נתיב:** globe-focusPlaceQuiet
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
פאנל גלובוס: focusPlaceQuiet
```

### 41. G03 — ✅ הצג את פריז

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
הצגתי לך את פריז על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 42. G04 — ✅ הצג את הר האוורסט

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
הצגתי לך את הר האוורסט על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 43. G05 — ✅ הצג את תעלת פנמה

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
הצגתי לך את תעלת פנמה על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 44. G06 — ✅ הצג את משולש ברמודה

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
הצגתי לך את משולש ברמודה על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 45. T01 — ⏭️ מהי תחנת הרכבת הקרובה ביותר לשדה התעופה בברלין?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 46. T02 — ✅ הצג אותה על המפה

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
הצגתי לך את ותה על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 47. T03 — ⏭️ כמה זמן נסיעה משם למרכז העיר?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 48. T04 — ⏭️ אילו קווי רכבת מגיעים לשם?

- **נתיב:** no-search
- **Intents:** —
- **מקורות:** —
- **זמן:** 0ms

```
לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.
```

### 49. GH01 — ✅ מצא פרויקטי WebGPU חדשים

- **נתיב:** canned-live-reply
- **Intents:** github
- **מקורות:** github
- **זמן:** 580ms

```
לפי GitHub:
• שאילתה: WebGPU
- mrdoob/three.js [JavaScript]: JavaScript 3D Library. (https://github.com/mrdoob/three.js) ★113029
- BabylonJS/Babylon.js [TypeScript]: Babylon.js is a powerful, beautiful, simple, and open game and rendering engine packed into a friendly JavaScript framework. (https://github.com/BabylonJS/Babylon.js) ★25633
- mlc-ai/web-llm [TypeScript]: High-performance In-browser LLM Inference Engine  (https://github.com/mlc-ai/web-llm) ★18186
- gfx-rs/wgpu [Rust]: A cross-platform, safe, pure-Rust graphics API. (https://github.com/gfx-rs/wgpu) ★17350
- playcanvas/engine [JavaScript]: Powerful web graphics runtime built on WebGL, WebGPU, WebXR and glTF (https://github.com/playcanvas/engine) ★16021
- tracel-ai/burn [Rust]: Burn is a next generation tensor library and Deep Learning Framework that doesn't compromise on flexibility, efficiency and portability. (https://github.com/tracel-ai/burn) ★15416
מקור: GitHub Repositories.
```

### 50. GH02 — ✅ מצא פרויקטי AI שפורסמו השבוע

- **נתיב:** canned-live-reply
- **Intents:** github
- **מקורות:** github
- **זמן:** 498ms

```
לפי GitHub:
• שאילתה: AI stars:>50 pushed:>2025-01-01
- openclaw/openclaw [TypeScript]: Your own personal AI assistant. Any OS. Any Platform. The lobster way. 🦞  (https://github.com/openclaw/openclaw) ★378569
- NousResearch/hermes-agent [Python]: The agent that grows with you (https://github.com/NousResearch/hermes-agent) ★192754
- n8n-io/n8n [TypeScript]: Fair-code workflow automation platform with native AI capabilities. Combine visual building with custom code, self-host or cloud, 400+ integrations. (https://github.com/n8n-io/n8n) ★192389
- Significant-Gravitas/AutoGPT [Python]: AutoGPT is the vision of accessible AI for everyone, to use and to build on. Our mission is to provide the tools, so that you can focus on what matters. (https://github.com/Significant-Gravitas/AutoGPT) ★184930
- AUTOMATIC1111/stable-diffusion-webui [Python]: Stable Diffusion web UI (https://github.com/AUTOMATIC1111/stable-diffusion-webui) ★163677
- f/prompts.chat [HTML]: f.k.a. Awesome ChatGPT Prompts. Share, discover, and collect prompts from the community. Free and open source — self-host for your organization with complete privacy. (https://github.com/f/prompts.chat) ★163675
מקור: GitHub Repositories.
```

### 51. GH03 — ✅ מהם הפרויקטים הפופולריים ביותר היום?

- **נתיב:** canned-live-reply
- **Intents:** hackernews, github
- **מקורות:** github, hacker-news
- **זמן:** 1461ms

```
לפי GitHub:
• שאילתה: stars:>100 pushed:>2024-01-01
- codecrafters-io/build-your-own-x [Markdown]: Master programming by recreating your favorite technologies from scratch. (https://github.com/codecrafters-io/build-your-own-x) ★515078
- sindresorhus/awesome: 😎 Awesome lists about all kinds of interesting topics (https://github.com/sindresorhus/awesome) ★475502
- freeCodeCamp/freeCodeCamp [TypeScript]: freeCodeCamp.org's open-source codebase and curriculum. Learn math, programming, and computer science for free. (https://github.com/freeCodeCamp/freeCodeCamp) ★446824
- public-apis/public-apis [Python]: A collective list of free APIs (https://github.com/public-apis/public-apis) ★441339
- EbookFoundation/free-programming-books [Python]: :books: Freely available programming books (https://github.com/EbookFoundation/free-programming-books) ★390191
- openclaw/openclaw [TypeScript]: Your own personal AI assistant. Any OS. Any Platform. The lobster way. 🦞  (https://github.com/openclaw/openclaw) ★378569
מקור: GitHub Repositories.
```

### 52. GH04 — ⚠️ מצא משחקים שנבנו עם Three.js

- **נתיב:** game-panel + archive.org
- **Intents:** github
- **מקורות:** —
- **זמן:** 2989ms

```
לא מצאתי משחק שתואם ל«נבנו עם Three.js». אפשר לחפש לפי קטגוריה — בחר מהרשימה למטה (ארקייד, PS1, מירוצים, מומלצים ועוד). אותן קטגוריות מופיעות גם בחלון המשחקים בצד ימין.
```

### 53. GH05 — ✅ מצא חלופות ל-Ollama

- **נתיב:** canned-live-reply
- **Intents:** github
- **מקורות:** github
- **זמן:** 497ms

```
לפי GitHub:
• שאילתה: Ollama
- ollama/ollama [Go]: Get up and running with Kimi-K2.6, GLM-5.1, MiniMax, DeepSeek, gpt-oss, Qwen, Gemma and other models. (https://github.com/ollama/ollama) ★174070
- open-webui/open-webui [Python]: User-friendly AI Interface (Supports Ollama, OpenAI API, ...) (https://github.com/open-webui/open-webui) ★141393
- ChatGPTNextWeb/NextChat [TypeScript]: ✨ Light and Fast AI Assistant. Support: Web | iOS | MacOS | Android |  Linux | Windows (https://github.com/ChatGPTNextWeb/NextChat) ★88241
- chatboxai/chatbox [TypeScript]: Powerful AI Client (https://github.com/chatboxai/chatbox) ★40448
- chatchat-space/Langchain-Chatchat [Python]: Langchain-Chatchat（原Langchain-ChatGLM）基于 Langchain 与 ChatGLM, Qwen 与 Llama 等语言模型的 RAG 与 Agent 应用 | Langchain-Chatchat (formerly langchain-ChatGLM), local knowledge based LLM (like ChatGLM, Qwen and Llama) RAG and Agent app with langchain  (https://github.com/chatchat-space/Langchain-Chatchat) ★38168
- 1Panel-dev/1Panel [Go]: 🔥 1Panel is a modern, open-source VPS control panel — and the only one with native AI agent support. Run Ollama models, deploy OpenClaw agents, and manage your entire server stack from one clean web interface. (https://github.com/1Panel-dev/1Panel) ★35877
מקור: GitHub Repositories.
```

### 54. HF01 — ✅ מצא מודלים חדשים השבוע

- **נתיב:** canned-live-reply
- **Intents:** github, huggingface
- **מקורות:** github, huggingface-models, huggingface-datasets
- **זמן:** 1115ms

```
לפי GitHub:
• שאילתה: llm language model
- mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. (https://github.com/mlabonne/llm-course) ★80098
- sgl-project/sglang [Python]: SGLang is a high-performance serving framework for large language models and multimodal models. (https://github.com/sgl-project/sglang) ★28968
- HandsOnLLM/Hands-On-Large-Language-Models [Jupyter Notebook]: Official code repo for the O'Reilly Book - "Hands-On Large Language Models" (https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) ★26974
- Hannibal046/Awesome-LLM: Awesome-LLM: a curated list of Large Language Model (https://github.com/Hannibal046/Awesome-LLM) ★26928
- yamadashy/repomix [TypeScript]: 📦 Repomix is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more. (https://github.com/yamadashy/repomix) ★26245
- p-e-w/heretic [Python]: Fully automatic censorship removal for language models (https://github.com/p-e-w/heretic) ★24432
מקור: GitHub Repositories.
```

### 55. HF02 — ✅ מהם מודלי ה-VLM הפופולריים ביותר?

- **נתיב:** canned-live-reply
- **Intents:** huggingface
- **מקורות:** huggingface-models, huggingface-datasets
- **זמן:** 175ms

```
לפי Hugging Face (מודלים):
• שאילתה: vision-language
- NovaSearch/jasper_en_vision_language_v1 (sentence-transformers, safetensors, jasper_vl) · ⬇399 · ♥70
•   https://huggingface.co/NovaSearch/jasper_en_vision_language_v1
- NOVA-vision-language/GlorIA-1.3B (text-generation) · ⬇348 · ♥34
•   https://huggingface.co/NOVA-vision-language/GlorIA-1.3B
- LeroyDyer/Language_VisionModel_GGUF (image-text-to-text) · ⬇39 · ♥2
•   https://huggingface.co/LeroyDyer/Language_VisionModel_GGUF
- alexlimh/jasper_en_vision_language_v1 (sentence-transformers, safetensors, jasper_vl) · ⬇18 · ♥0
•   https://huggingface.co/alexlimh/jasper_en_vision_language_v1
- NOVA-vision-language/polite_bert (text-classification) · ⬇15 · ♥3
•   https://huggingface.co/NOVA-vision-language/polite_bert
- Joe99/visionlanguageTransformer (visual-question-answering) · ⬇9 · ♥0
•   https://huggingface.co/Joe99/visionlanguageTransformer
מקור: Hugging Face Models.
```

### 56. HF03 — ✅ מצא מודלים לזיהוי אובייקטים

- **נתיב:** canned-live-reply
- **Intents:** github, huggingface
- **מקורות:** github, huggingface-models, huggingface-datasets
- **זמן:** 176ms

```
לפי GitHub:
• שאילתה: llm language model
- mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. (https://github.com/mlabonne/llm-course) ★80098
- sgl-project/sglang [Python]: SGLang is a high-performance serving framework for large language models and multimodal models. (https://github.com/sgl-project/sglang) ★28968
- HandsOnLLM/Hands-On-Large-Language-Models [Jupyter Notebook]: Official code repo for the O'Reilly Book - "Hands-On Large Language Models" (https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) ★26974
- Hannibal046/Awesome-LLM: Awesome-LLM: a curated list of Large Language Model (https://github.com/Hannibal046/Awesome-LLM) ★26928
- yamadashy/repomix [TypeScript]: 📦 Repomix is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more. (https://github.com/yamadashy/repomix) ★26245
- p-e-w/heretic [Python]: Fully automatic censorship removal for language models (https://github.com/p-e-w/heretic) ★24432
מקור: GitHub Repositories.
```

### 57. HF04 — ✅ מצא מודלים לזיהוי תנוחות גוף

- **נתיב:** canned-live-reply
- **Intents:** github, huggingface
- **מקורות:** github, huggingface-models, huggingface-datasets
- **זמן:** 157ms

```
לפי GitHub:
• שאילתה: llm language model
- mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. (https://github.com/mlabonne/llm-course) ★80098
- sgl-project/sglang [Python]: SGLang is a high-performance serving framework for large language models and multimodal models. (https://github.com/sgl-project/sglang) ★28968
- HandsOnLLM/Hands-On-Large-Language-Models [Jupyter Notebook]: Official code repo for the O'Reilly Book - "Hands-On Large Language Models" (https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) ★26974
- Hannibal046/Awesome-LLM: Awesome-LLM: a curated list of Large Language Model (https://github.com/Hannibal046/Awesome-LLM) ★26928
- yamadashy/repomix [TypeScript]: 📦 Repomix is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more. (https://github.com/yamadashy/repomix) ★26245
- p-e-w/heretic [Python]: Fully automatic censorship removal for language models (https://github.com/p-e-w/heretic) ★24432
מקור: GitHub Repositories.
```

### 58. HF05 — ✅ מצא מודלים ל-WebGPU

- **נתיב:** canned-live-reply
- **Intents:** github, huggingface
- **מקורות:** github, huggingface-models, huggingface-datasets
- **זמן:** 159ms

```
לפי GitHub:
• שאילתה: WebGPU
- mrdoob/three.js [JavaScript]: JavaScript 3D Library. (https://github.com/mrdoob/three.js) ★113029
- BabylonJS/Babylon.js [TypeScript]: Babylon.js is a powerful, beautiful, simple, and open game and rendering engine packed into a friendly JavaScript framework. (https://github.com/BabylonJS/Babylon.js) ★25633
- mlc-ai/web-llm [TypeScript]: High-performance In-browser LLM Inference Engine  (https://github.com/mlc-ai/web-llm) ★18186
- gfx-rs/wgpu [Rust]: A cross-platform, safe, pure-Rust graphics API. (https://github.com/gfx-rs/wgpu) ★17350
- playcanvas/engine [JavaScript]: Powerful web graphics runtime built on WebGL, WebGPU, WebXR and glTF (https://github.com/playcanvas/engine) ★16021
- tracel-ai/burn [Rust]: Burn is a next generation tensor library and Deep Learning Framework that doesn't compromise on flexibility, efficiency and portability. (https://github.com/tracel-ai/burn) ★15416
מקור: GitHub Repositories.
```

### 59. HF06 — ✅ מצא מודלים שמתאימים להרצה בדפדפן

- **נתיב:** canned-live-reply
- **Intents:** github, huggingface
- **מקורות:** github
- **זמן:** 160ms

```
לפי GitHub:
• שאילתה: llm language model
- mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. (https://github.com/mlabonne/llm-course) ★80098
- sgl-project/sglang [Python]: SGLang is a high-performance serving framework for large language models and multimodal models. (https://github.com/sgl-project/sglang) ★28968
- HandsOnLLM/Hands-On-Large-Language-Models [Jupyter Notebook]: Official code repo for the O'Reilly Book - "Hands-On Large Language Models" (https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) ★26974
- Hannibal046/Awesome-LLM: Awesome-LLM: a curated list of Large Language Model (https://github.com/Hannibal046/Awesome-LLM) ★26928
- yamadashy/repomix [TypeScript]: 📦 Repomix is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more. (https://github.com/yamadashy/repomix) ★26245
- p-e-w/heretic [Python]: Fully automatic censorship removal for language models (https://github.com/p-e-w/heretic) ★24432
מקור: GitHub Repositories.
```

### 60. GM01 — ✅ שחק Doom

- **נתיב:** game-panel + archive.org
- **Intents:** —
- **מקורות:** archive.org
- **זמן:** 1335ms

```
נמצאו 8 משחקים התואמים ל«doom». הם מוצגים בחלון המשחקים שנפתח בצד ימין — בחר כרטיס ולחץ ▶ שחק עכשיו.
```

### 61. GM02 — ✅ שחק Doom II

- **נתיב:** game-panel + archive.org
- **Intents:** —
- **מקורות:** archive.org
- **זמן:** 661ms

```
נמצאו 1 משחקים התואמים ל«doom 2». הם מוצגים בחלון המשחקים שנפתח בצד ימין — בחר כרטיס ולחץ ▶ שחק עכשיו.
```

### 62. GM03 — ✅ שחק Dune II

- **נתיב:** game-panel + archive.org
- **Intents:** —
- **מקורות:** archive.org
- **זמן:** 535ms

```
נמצאו 1 משחקים התואמים ל«dune 2». הם מוצגים בחלון המשחקים שנפתח בצד ימין — בחר כרטיס ולחץ ▶ שחק עכשיו.
```

### 63. GM04 — ✅ שחק Prince of Persia

- **נתיב:** game-panel + archive.org
- **Intents:** —
- **מקורות:** archive.org
- **זמן:** 6261ms

```
נמצאו 7 משחקים התואמים ל«prince of persia». הם מוצגים בחלון המשחקים שנפתח בצד ימין — בחר כרטיס ולחץ ▶ שחק עכשיו.
```

### 64. GM05 — ✅ שחק Wolfenstein 3D

- **נתיב:** game-panel + archive.org
- **Intents:** —
- **מקורות:** archive.org
- **זמן:** 1739ms

```
נמצאו 8 משחקים התואמים ל«wolfenstein». הם מוצגים בחלון המשחקים שנפתח בצד ימין — בחר כרטיס ולחץ ▶ שחק עכשיו.
```

### 65. GM06 — ✅ מצא משחקי DOS אסטרטגיה

- **נתיב:** game-panel + archive.org
- **Intents:** github
- **מקורות:** archive.org
- **זמן:** 3441ms

```
נמצאו 8 משחקים בקטגוריית «משחקים: PC / DOS». הם מוצגים בחלון המשחקים בצד ימין — בחר משחק ולחץ ▶ שחק עכשיו.
```

### 66. GM07 — ⚠️ מצא משחקי SEGA משנות ה-90

- **נתיב:** game-panel + archive.org
- **Intents:** github
- **מקורות:** —
- **זמן:** 1708ms

```
לא מצאתי משחק שתואם ל«י SEGA משנות ה-90». אפשר לחפש לפי קטגוריה — בחר מהרשימה למטה (ארקייד, PS1, מירוצים, מומלצים ועוד). אותן קטגוריות מופיעות גם בחלון המשחקים בצד ימין.
```

### 67. F01 — ✅ כמה מטוסים נמצאים מעל האזור שבו נמצאת כרגע הסופה הגדולה בעול

- **נתיב:** canned-live-reply
- **Intents:** marine, aviation
- **מקורות:** adsb-aviation
- **זמן:** 827ms

```
לפי ADS-B חי:
• אזור: גלובלי (OpenSky ADS-B)
• מטוסים באוויר (דיווח אחרון): 7327
• הערה: עולם חי (🌐) מציג מטוסים על המפה — «הצג על המפה» לצפייה.
1. N36VK · גובה 3185.16m · 65m/s
2. TRF559 · גובה 769.62m · 57m/s
3. DAL2695 · גובה 10668m · 251m/s
4. ASA406 · גובה 10058.4m · 276m/s
5. CARBN22 · גובה 3512.82m · 219m/s
6. N5642M · גובה 457.2m · 47m/s
7. IGO605V · גובה 3604.26m · 184m/s
8. AXB2795 · גובה 9966.96m · 222m/s
9. N7198E · גובה 304.8m · 63m/s
10. AAL2028 · גובה 3070.86m · 180m/s
מקור: תעופה (ADS-B).
```

### 68. F02 — ✅ האם יש אוניות באזורי התרעת צונאמי?

- **נתיב:** canned-live-reply
- **Intents:** ships, disaster
- **מקורות:** ais-ships, gdacs-disasters
- **זמן:** 1804ms

```
לפי נתוני עולם חי / Digitraffic סביב גלובלי (Digitraffic):
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
דוגמאות:
• MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
• MMSI 376128000 · AIS · 59.18,19.61 · 0.7 kn
• MMSI 273274000 · AIS · 59.89,30.19 · 0.5 kn
• MMSI 231319000 · AIS · 56.05,17.60 · 0.2 kn
מקור: עולם חי (AIS / סימוני מסלול).
```

### 69. F03 — ✅ האם תחנת החלל נמצאת כרגע מעל מדינה שבה יש סערה משמעותית?

- **נתיב:** canned-live-reply
- **Intents:** marine, satellite, country
- **מקורות:** iss-tracker
- **זמן:** 5764ms

```
לפי מעקב ISS:
• מיקום ISS (זמן אמת):
• קו רוחב: -45.48°
• קו אורך: -76.23°
• גובה: 433 km
• מהירות: 27550 km/h
מקור: תחנת חלל (ISS / עולם חי).
```

### 70. F04 — ✅ הצג את כל רעידות האדמה שהתרחשו בטווח של 500 ק"מ מנתיבי שיט ר

- **נתיב:** canned-live-reply
- **Intents:** earthquake
- **מקורות:** usgs-earthquake
- **זמן:** 1099ms

```
לפי USGS (רעידות אדמה):
• אין רעידות אדמה מדווחות ב-24 שעות באזור שהתרחשו, בטווח, מנתיבי (USGS).
• לא נמצאו רעידות באזור (שהתרחשו, בטווח, מנתיבי) ב-24 שעות.
מקור: רעידות אדמה (USGS).
```

### 71. F05 — ✅ אילו שדות תעופה נמצאים במסלול של סופת הוריקן פעילה?

- **נתיב:** canned-live-reply
- **Intents:** disaster
- **מקורות:** gdacs-disasters
- **זמן:** 942ms

```
לפי GDACS (אסונות טבע):
• אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green
4.  · Iceland · Green
5.  · Mexico · Green
6.  · Italy · Green
7.  · Philippines · Green
8.  · Indonesia · Green
מקור: אסונות (GDACS).
```

### 72. O01 — ✅ מה הדברים המעניינים שקורים בעולם עכשיו?

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10011ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
סה"כ 198 רעידות ב-24 שעות (USGS). 8 הגדולות:
- M5.2 · 2 km WSW of Kablalan, Philippines · 2026-06-13 02:05:53 UTC
  https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssls
- M5.1 · 2 km S of Cerro de Piedra, Mexico · 2026-06-13 18:20:42 UTC
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:11 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: גלובלי (OpenSky ADS-B)
מטוסים באוויר (דיווח אחרון): 7332
הערה: עולם חי (🌐) מציג מטוסים על המפה — «הצג על המפה» לצפייה.
1. N36VK · גובה 3230.88m · 67m/s
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 73. O02 — ✅ האם יש משהו חריג שמתרחש כרגע?

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10015ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור משהו, חריג, שמתרחש (USGS).
לא נמצאו רעידות באזור (משהו, חריג, שמתרחש) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:11 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 634
1. N445ME · גובה 2150ft · 120kn
2. N512TR · גובה 1800ft · 86kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 74. O03 — ✅ אילו אירועים חשובים התרחשו ב-24 השעות האחרונות?

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10013ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור אילו, אירועים, חשובים (USGS).
לא נמצאו רעידות באזור (אילו, אירועים, חשובים) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:12 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 632
1. N445ME · גובה 2150ft · 120kn
2. UAL1776 · גובה 34200ft · 517kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 75. O04 — ✅ תן לי סקירה של מצב העולם כרגע

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** open-meteo, open-meteo-marine, usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10019ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**מזג אוויר (Open-Meteo)**
מיקום: העולם, Dubai, AE
גובה: 9999 m
זמן (מקומי): 2026-06-14T04:00
מצב: סופת רעמים
**ים וגלים (Open-Meteo Marine)**
מיקום: העולם, Dubai, AE
זמן: 2026-06-14T04:00
גובה גל: 0.3 m
כיוון גל: 299°
**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור סקירה, העולם, כרגע (USGS).
לא נמצאו רעידות באזור (סקירה, העולם, כרגע) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: העולם
ספינות בטווח: 1 (0 AIS חי + 1 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. Dubai Jebel Ali · מסלול · 25.01,55.06 · — → AEJEA
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:12 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 625
1. N445ME · גובה 2150ft · 120kn
2. N512TR · גובה 2200ft · 89kn

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 76. O05 — ✅ הצג לי את המקומות הפעילים ביותר על פני כדור הארץ כרגע

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10011ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור המקומות, הפעילים, ביותר (USGS).
לא נמצאו רעידות באזור (המקומות, הפעילים, ביותר) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:12 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 633
1. N445ME · גובה 2150ft · 120kn
2. N512TR · גובה 2300ft · 98kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 77. O06 — ✅ מה קורה עכשיו בחלל?

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10023ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור קורה, בחלל (USGS).
לא נמצאו רעידות באזור (קורה, בחלל) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:12 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 633
1. N445ME · גובה 2150ft · 121kn
2. N512TR · גובה 2400ft · 116kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 78. O07 — ✅ מה קורה עכשיו באוקיינוסים?

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10016ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור קורה, באוקיינוסים (USGS).
לא נמצאו רעידות באזור (קורה, באוקיינוסים) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:12 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 637
1. N445ME · גובה 2150ft · 121kn
2. N512TR · גובה 2500ft · 127kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 79. O08 — ✅ מה קורה עכשיו בשמי אירופה?

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10021ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור קורה, בשמי, אירופה (USGS).
לא נמצאו רעידות באזור (קורה, בשמי, אירופה) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:12 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: מרכז אירופה · רדיוס 250km
מטוסים בטווח: 105
1. TOM12Y · גובה 38000ft · 366kn
2. EXS21LK · גובה 36000ft · 361kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 80. ST01 — ✅ תן לי תמונת מצב מלאה של כדור הארץ כרגע

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10009ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור תמונת, מלאה, כדור (USGS).
לא נמצאו רעידות באזור (תמונת, מלאה, כדור) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:13 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 634
1. UAL1776 · גובה 33125ft · 517kn
2. GJS4432 · גובה 21000ft · 420kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 81. ST02 — ✅ מה 20 האירועים החריגים ביותר שמתרחשים עכשיו?

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10015ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
אין רעידות אדמה מדווחות ב-24 שעות באזור האירועים, החריגים, ביותר (USGS).
לא נמצאו רעידות באזור (האירועים, החריגים, ביותר) ב-24 שעות.
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:13 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: ברירת מחדל (NYC) · רדיוס 250km
מטוסים בטווח: 628
1. UAL1776 · גובה 32950ft · 517kn
2. AAL2689 · גובה 26000ft · 401kn
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 82. ST03 — ✅ הצג על הגלובוס בו זמנית מטוסים, אוניות, רעידות אדמה, סופות ו

- **נתיב:** canned-live-reply
- **Intents:** ships, earthquake, aviation, disaster
- **מקורות:** usgs-earthquake, ais-ships, adsb-aviation, gdacs-disasters
- **זמן:** 1210ms

```
לפי נתוני עולם חי / Digitraffic סביב גלובלי (Digitraffic):
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
דוגמאות:
• MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
• MMSI 376128000 · AIS · 59.18,19.61 · 0.7 kn
• MMSI 273274000 · AIS · 59.89,30.19 · 0.5 kn
• MMSI 231319000 · AIS · 56.05,17.60 · 0.2 kn
מקור: עולם חי (AIS / סימוני מסלול).
```

### 83. ST04 — ✅ סכם את כל ההתראות הפעילות בעולם

- **נתיב:** canned-live-reply
- **Intents:** disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts
- **מקורות:** usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news
- **זמן:** 10021ms

```
סקירת מצב עולם (נתונים חיים ממקורות מרובים):

**רעידות אדמה (USGS)**
סה"כ 198 רעידות ב-24 שעות (USGS). 8 הגדולות:
- M5.2 · 2 km WSW of Kablalan, Philippines · 2026-06-13 02:05:53 UTC
  https://earthquake.usgs.gov/earthquakes/eventpage/us7000ssls
- M5.1 · 2 km S of Cerro de Piedra, Mexico · 2026-06-13 18:20:42 UTC
**ספינות (AIS / עולם חי)**
אזור: גלובלי (Digitraffic)
ספינות בטווח: 18559 (18521 AIS חי + 38 סימוני מסלול + 0 עולם חי)
הערה: Digitraffic מכסה בעיקר צפון אירופה; באזורים ללא AIS חי מוצגים סימוני מסלול מעולם חי.
1. MMSI 219598000 · AIS · 55.77,20.85 · 0.0 kn
**חדשות (RSS)**
מקור: BBC News
כותרות עדכניות (2026-06-14T00:13 UTC):
1. Trump says US-Iran deal to be signed on Sunday as Tehran casts doubt on timing
2. Resident doctors cancel strike after new offer from government
**תעופה (ADS-B)**
אזור: גלובלי (OpenSky ADS-B)
מטוסים באוויר (דיווח אחרון): 7329
הערה: עולם חי (🌐) מציג מטוסים על המפה — «הצג על המפה» לצפייה.
1. N8180F · גובה 762m · 62m/s
**התרעות ישראל (צבע אדום)**
✅ אין התרעות פעילות כרגע בישראל
**אסונות (GDACS)**
אירועי טבע (GDACS):
1.  · Türkiye · Green
2.  · Philippines · Green
3.  · Philippines · Green

הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.
```

### 84. L01 — ✅ הצג רעידות אדמה על הגלובוס

- **נתיב:** canned-live-reply
- **Intents:** earthquake
- **מקורות:** usgs-earthquake
- **זמן:** 58ms

```
לפי USGS (רעידות אדמה):
• אין רעידות אדמה מדווחות ב-24 שעות באזור הגלובוס (USGS).
• לא נמצאו רעידות באזור (הגלובוס) ב-24 שעות.
מקור: רעידות אדמה (USGS).
```

### 85. L02 — ✅ מה מהירות הרוח בפריז?

- **נתיב:** canned-live-reply
- **Intents:** weather
- **מקורות:** open-meteo
- **זמן:** 329ms

```
לפי Open-Meteo (מזג אוויר):
• מיקום: Paris, FR
• זמן (מקומי): 2026-06-14T02:00
• מצב: בהיר ברובו
• טמפרatura: 18°C (מרגיש 16.3°C)
• לחות: 54%
• רוח: 9.4 km/h, כיוון 317°
• לחץ: 1016.4 hPa
• תחזית 3 ימים:
- 2026-06-14 (2026-06-14): 14.7–25.9°C, גשם 0 mm (סיכוי 0%) · מעונן, רוח עד 12 km/h
- 2026-06-15 (2026-06-15): 15.3–27.8°C, גשם 0 mm (סיכוי 5%) · מעונן, רוח עד 9.9 km/h
- 2026-06-16 (2026-06-16): 19.9–26.7°C, גשם 0 mm (סיכוי 5%) · מעונן, רוח עד 10.9 km/h
מקור: מזג אוויר (Open-Meteo).
```

### 86. L03 — ✅ מתי תחנת החלל תעבור מעל ישראל?

- **נתיב:** canned-live-reply
- **Intents:** satellite
- **מקורות:** iss-tracker
- **זמן:** 14037ms

```
לפי מעקב ISS:
• מיקום ISS (זמן אמת):
• קו רוחב: -48.89°
• קו אורך: -65.53°
• גובה: 435 km
• מהירות: 27545 km/h
מקור: תחנת חלל (ISS / עולם חי).
```

### 87. L04 — ✅ הצג על המפה את גרמניה

- **נתיב:** globe-place (canned)
- **Intents:** —
- **מקורות:** —
- **זמן:** 1ms

```
הצגתי לך את על המפה את גרמניה על המפה בפאנל REALITY LIVE מימין — המפה ממוקדת שם עם סימון מהבהב. אפשר להדליק שכבות (מטוסים, מזג אוויר וכו') מהכפתורים למעלה אם תרצה.
```

### 88. L05 — ✅ כמה ק"מ בין ירושלים לחיפה?

- **נתיב:** globe-focusPlaceQuiet
- **Intents:** distance
- **מקורות:** —
- **זמן:** 0ms

```
פאנל גלובוס: focusPlaceQuiet
```

### 89. L06 — ✅ חפש פרויקטים בנושא WebGPU

- **נתיב:** canned-live-reply
- **Intents:** github, wikipedia
- **מקורות:** github, wikipedia-he
- **זמן:** 686ms

```
לפי GitHub:
• שאילתה: WebGPU
- mrdoob/three.js [JavaScript]: JavaScript 3D Library. (https://github.com/mrdoob/three.js) ★113029
- BabylonJS/Babylon.js [TypeScript]: Babylon.js is a powerful, beautiful, simple, and open game and rendering engine packed into a friendly JavaScript framework. (https://github.com/BabylonJS/Babylon.js) ★25633
- mlc-ai/web-llm [TypeScript]: High-performance In-browser LLM Inference Engine  (https://github.com/mlc-ai/web-llm) ★18186
- gfx-rs/wgpu [Rust]: A cross-platform, safe, pure-Rust graphics API. (https://github.com/gfx-rs/wgpu) ★17350
- playcanvas/engine [JavaScript]: Powerful web graphics runtime built on WebGL, WebGPU, WebXR and glTF (https://github.com/playcanvas/engine) ★16021
- tracel-ai/burn [Rust]: Burn is a next generation tensor library and Deep Learning Framework that doesn't compromise on flexibility, efficiency and portability. (https://github.com/tracel-ai/burn) ★15416
מקור: GitHub Repositories.
```

### 90. L07 — ✅ כמה שקלים שווים 100 דולר?

- **נתיב:** canned-live-reply
- **Intents:** currency
- **מקורות:** frankfurter-fx
- **זמן:** 2012ms

```
לפי שערי מטבע (Frankfurter):
• תאריך: 2026-06-12
• 100 USD = 292.07 ILS
• 1 USD = 2.9207 ILS
• 100 USD = 292.0700 ILS
• מקור: European Central Bank via Frankfurter
מקור: שערי מטבע (Frankfurter).
```

### 91. L08 — ✅ כמה תושבים יש בקנדה?

- **נתיב:** globe-focusPlaceQuiet
- **Intents:** country
- **מקורות:** —
- **זמן:** 1ms

```
פאנל גלובוס: focusPlaceQuiet
```

### 92. L09 — ✅ מה מחיר הביטקוין עכשיו?

- **נתיב:** canned-live-reply
- **Intents:** crypto
- **מקורות:** coingecko
- **זמן:** 352ms

```
לפי CoinGecko (קריפטו):
• bitcoin: $64384 USD
• ≈ ₪188,061
• שינוי 24h: 1.31%
מקור: CoinGecko (קריפטו).
```

