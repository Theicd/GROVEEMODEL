# דוח בדיקות יכולות GROVEE — 2026-06-14

## סיכום

| סטטוס | כמות |
|--------|------|
| ✅ עובד (מקורות/Intent) | 54 |
| ⚠️ חלקי | 14 |
| ❌ נכשל | 11 |
| 🔵 ידני / LLM בלבד | 0 |
| ⏭️ לא נתמך (צפוי) | 4 |

**סה"כ שאלות:** 83

## מפת יכולות (לפי קטגוריה)

- **אוניות** — 5/6 עובר
- **חדשות** — 6/7 עובר
- **חלל** — 2/5 עובר
- **ים** — 1/2 עובר
- **מזג** — 2/3 עובר
- **מפות** — 6/6 עובר
- **משחקים** — 7/7 עובר
- **סקירה** — 2/8 עובר
- **צבא** — 3/4 עובר
- **רעידות** — 4/5 עובר
- **שילוב** — 0/5 עובר
- **תחבורה** — 1/4 עובר
- **תעופה** — 3/5 עובר
- **תשתיות ים** — 0/1 עובר
- **GitHub** — 5/5 עובר
- **Hugging Face** — 6/6 עובר
- **stress** — 1/4 עובר

## מה חוזר על עצמו (דפוסים)

| דפוס | המלצה |
|------|--------|
| «הצג על הגלובוס/מפה» | ✅ Globe intent — עובד כשיש מקום/מדינה מפורש |
| «כמה X באזור Y» (מטוסים/אוניות) | ✅ כש-Y במאגר bbox (ישראל, חיפה, רוטרדם) — ⚠️ לונדון/סואץ חלש |
| «הכי עמוס / הכי גדול / Starlink» | ❌ אין API דירוג real-time |
| שילוב 2+ מקורות (סופה+מטוסים) | 🔵 דורש LLM — חיפוש לא משלב לבד |
| «שחק X» | ✅ Game panel — חלק מהניסוחים («שחק Doom») צריכים חידוד |
| סקירת עולם / 20 אירועים | 🔵 LLM + partial search — לא תשובה מובנית אחת |
| GitHub / HF / מזג / רעידות / ISS | ✅ מקורות חיים — תשובה תלויה ב-Gemma |

## פירוט לפי שאלה

### אוניות

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| S01 | כמה אוניות נמצאות במפרץ סואץ? | ✅ | ships | ais-ships | אזור: תעלת סואץ |
| S02 | כמה מכליות נפט נמצאות במפרץ הפרסי? | ✅ | ships | ais-ships | אזור: מפרץ הפרסי |
| S03 | אילו אוניות נמצאות ליד חופי ישראל? | ✅ | ships, places | ais-ships | אזור: ישראל (חוף) |
| S04 | הצג אוניות מכולה באזור רוטרדם | ✅ | ships | ais-ships | אזור: נמל רוטרדם |
| S05 | מהו הנמל העמוס ביותר כרגע? | ⏭️ | — | — | צפוי — אין מקור |
| S06 | כמה כלי שייט או אוניות יש במפרץ חיפה? | ✅ | ships | ais-ships | אזור: מפרץ חיפה |

### חדשות

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| N01 | מה הכותרות החשובות בעולם כרגע? | ❌ | — | — | חלש/חלקי |
| N02 | מה קורה בישראל היום? | ✅ | news | news-rss | מקור: ynet |
| N03 | האם יש אסונות טבע פעילים כרגע? | ✅ | disaster | gdacs-disasters | אירועי טבע (GDACS): |
| N04 | אילו מדינות נמצאות תחת התרעות מזג אוויר? | ✅ | weather, alerts, country | israel-alerts | ✅ אין התרעות פעילות כרגע בישראל |
| N05 | האם התרחשו רעידות אדמה משמעותיות ב-24 השעות האחרונות? | ✅ | earthquake | usgs-earthquake | אין רעידות אדמה מדווחות ב-24 שעות באזור התרחשו, משמעותיות, ב |
| N06 | מהי השריפה הגדולה ביותר הפעילה כרגע בעולם? | ✅ | disaster | gdacs-disasters | אירועי טבע (GDACS): |
| N07 | אילו סופות טרופיות פעילות כרגע? | ✅ | disaster | gdacs-disasters | אירועי טבע (GDACS): |

### חלל

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| SP01 | איפה נמצאת תחנת החלל כרגע? | ❌ | satellite | — | Timeout fetching https://api.open-notify.org/iss-now.json |
| SP02 | מתי היא תעבור מעל ישראל? | ✅ | satellite | iss-tracker | מיקום ISS (זמן אמת): |
| SP03 | אילו לווייני Starlink נמצאים מעל אירופה? | ⏭️ | — | — | אין Starlink feed |
| SP04 | כמה לוויינים פעילים במסלול נמוך? | ❌ | satellite | — | חלש/חלקי |
| SP05 | הצג את מסלול ה-ISS על הגלובוס | ✅ | satellite | — | globe: showLayer |

### ים

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| W01 | מה גובה הגלים מול חופי תל אביב? | ✅ | marine | open-meteo-marine | מיקום: Tel Aviv, Tel Aviv, IL |
| W02 | איפה נמצאים הגלים הגבוהים ביותר כרגע בעולם? | ❌ | marine | — | חלש/חלקי |

### מזג

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| W03 | מה הטמפרטורה באיסלנד עכשיו? | ✅ | weather | open-meteo | מיקום: איסלנד, IS |
| W04 | איזה אזור בעולם חווה את הרוחות החזקות ביותר כרגע? | ❌ | — | — | חלש/חלקי |
| W05 | הצג מפה של מזג האוויר באירופה | ✅ | weather | — | globe: showLayer |

### מפות

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| G01 | איפה נמצאת גרמניה? | ✅ | — | — | globe: focusPlaceQuiet |
| G02 | התקרב לברלין | ✅ | — | — | globe: focusPlaceQuiet |
| G03 | הצג את פריז | ✅ | — | — | globe: focusPlaceQuiet |
| G04 | הצג את הר האוורסט | ✅ | — | — | globe: focusPlaceQuiet |
| G05 | הצג את תעלת פנמה | ✅ | — | — | globe: focusPlaceQuiet |
| G06 | הצג את משולש ברמודה | ✅ | — | — | globe: focusPlaceQuiet |

### משחקים

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| GM01 | שחק Doom | ✅ | — | — | game query="doom" cat=dos |
| GM02 | שחק Doom II | ✅ | — | — | game query="doom 2" cat=dos |
| GM03 | שחק Dune II | ✅ | — | — | game query="dune 2" cat=dos |
| GM04 | שחק Prince of Persia | ✅ | — | — | game query="prince of persia" cat=dos |
| GM05 | שחק Wolfenstein 3D | ✅ | — | — | game query="wolfenstein" cat=dos |
| GM06 | מצא משחקי DOS אסטרטגיה | ✅ | github | — | game query="" cat=dos |
| GM07 | מצא משחקי SEGA משנות ה-90 | ✅ | github | — | game query="י SEGA משנות ה-90" cat=— |

### סקירה

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| O01 | מה הדברים המעניינים שקורים בעולם עכשיו? | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (7 מקורות) — דורש LLM לשילוב |
| O02 | האם יש משהו חריג שמתרחש כרגע? | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (7 מקורות) — דורש LLM לשילוב |
| O03 | אילו אירועים חשובים התרחשו ב-24 השעות האחרונות? | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (7 מקורות) — דורש LLM לשילוב |
| O04 | תן לי סקירה של מצב העולם כרגע | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | open-meteo, open-meteo-marine, usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (9 מקורות) — דורש LLM לשילוב |
| O05 | הצג לי את המקומות הפעילים ביותר על פני כדור הארץ כרגע | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (7 מקורות) — דורש LLM לשילוב |
| O06 | מה קורה עכשיו בחלל? | ✅ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | אין רעידות אדמה מדווחות ב-24 שעות באזור קורה, בחלל (USGS). |
| O07 | מה קורה עכשיו באוקיינוסים? | ✅ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | אין רעידות אדמה מדווחות ב-24 שעות באזור קורה, באוקיינוסים (U |
| O08 | מה קורה עכשיו בשמי אירופה? | ❌ | — | — | חלש/חלקי |

### צבא

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| M01 | כמה מטוסים צבאיים מזוהים כרגע באזור הים התיכון? | ✅ | aviation | adsb-aviation | אזור: ברירת מחדל (NYC) · רדיוס 250km |
| M02 | האם יש מטוסי תדלוק אווירי מעל אירופה? | ✅ | aviation | adsb-aviation | אזור: מרכז אירופה · רדיוס 250km |
| M03 | אילו מטוסי AWACS פעילים כרגע? | ⚠️ | aviation | adsb-aviation | unexpected data |
| M04 | הצג את מיקומם על המפה | ✅ | — | — | globe: focusPlaceQuiet |

### רעידות

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| E01 | הצג את 20 רעידות האדמה האחרונות בעולם | ✅ | earthquake | usgs-earthquake | סה"כ 202 רעידות ב-24 שעות (USGS). 8 הגדולות: |
| E02 | מה הייתה רעידת האדמה החזקה ביותר השבוע? | ✅ | earthquake | usgs-earthquake | סה"כ 2241 רעידות ב-7 ימים (USGS). 8 הגדולות: |
| E03 | הצג אותן על הגלובוס | ✅ | — | — | globe: focusPlaceQuiet |
| E04 | האם הייתה רעידת אדמה ליד יפן ב-48 השעות האחרונות? | ✅ | earthquake | usgs-earthquake | אין רעידות אדמה מדווחות ב-24 שעות באזור japan, honshu, hokka |
| E05 | כמה רעידות מעל 5.0 התרחשו החודש? | ❌ | — | — | USGS feed — לא סינון חודש מדויק |

### שילוב

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| F01 | כמה מטוסים נמצאים מעל האזור שבו נמצאת כרגע הסופה הגדולה | ⚠️ | marine, aviation | adsb-aviation | חיפוש חלקי (1 מקורות) — דורש LLM לשילוב |
| F02 | האם יש אוניות באזורי התרעת צונאמי? | ⚠️ | ships, disaster | ais-ships, gdacs-disasters | חיפוש חלקי (2 מקורות) — דורש LLM לשילוב |
| F03 | האם תחנת החלל נמצאת כרגע מעל מדינה שבה יש סערה משמעותית | ⚠️ | marine, satellite, country | iss-tracker | חיפוש חלקי (1 מקורות) — דורש LLM לשילוב |
| F04 | הצג את כל רעידות האדמה שהתרחשו בטווח של 500 ק"מ מנתיבי  | ⚠️ | earthquake | usgs-earthquake | חיפוש חלקי (1 מקורות) — דורש LLM לשילוב |
| F05 | אילו שדות תעופה נמצאים במסלול של סופת הוריקן פעילה? | ⚠️ | disaster | gdacs-disasters | חיפוש חלקי (1 מקורות) — דורש LLM לשילוב |

### תחבורה

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| T01 | מהי תחנת הרכבת הקרובה ביותר לשדה התעופה בברלין? | ❌ | — | — | חלש/חלקי |
| T02 | הצג אותה על המפה | ✅ | — | — | globe: focusPlaceQuiet |
| T03 | כמה זמן נסיעה משם למרכז העיר? | ❌ | — | — | חלש/חלקי |
| T04 | אילו קווי רכבת מגיעים לשם? | ⏭️ | — | — | GTFS לא מחובר |

### תעופה

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| A01 | כמה מטוסים נמצאים כרגע מעל ישראל? | ✅ | aviation | adsb-aviation | אזור: ישראל (מרכז) · רדיוס 250km |
| A02 | כמה מטוסים מעל לונדון? | ✅ | aviation | adsb-aviation | אזור: לונדון · רדיוס 250km |
| A03 | אילו טיסות מתקרבות לנחיתה בנתב"ג? | ❌ | — | — | חלש/חלקי |
| A04 | מהו שדה התעופה העמוס ביותר כרגע? | ⏭️ | — | — | אין API עמוסות real-time |
| A05 | הצג את כל המטוסים מעל הים התיכון | ✅ | aviation | — | globe: showLayer |

### תשתיות ים

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| S07 | כמה מצופים יש במפרץ חיפה? | ❌ | marine-infra | — | Error: HTTP 406 for https://overpass-api.de/api/interpreter |

### GitHub

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| GH01 | מצא פרויקטי WebGPU חדשים | ✅ | github | github | שאילתה: WebGPU |
| GH02 | מצא פרויקטי AI שפורסמו השבוע | ✅ | github | github | שאילתה: AI stars:>50 pushed:>2025-01-01 |
| GH03 | מהם הפרויקטים הפופולריים ביותר היום? | ✅ | hackernews, github | github, hacker-news | שאילתה: stars:>100 pushed:>2024-01-01 |
| GH04 | מצא משחקים שנבנו עם Three.js | ✅ | github | github | שאילתה: Three.js |
| GH05 | מצא חלופות ל-Ollama | ✅ | github | github | שאילתה: Ollama |

### Hugging Face

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| HF01 | מצא מודלים חדשים השבוע | ✅ | github, huggingface | github, huggingface-models, huggingface-datasets | שאילתה: llm language model |
| HF02 | מהם מודלי ה-VLM הפופולריים ביותר? | ✅ | huggingface | huggingface-models, huggingface-datasets | שאילתה: vision-language |
| HF03 | מצא מודלים לזיהוי אובייקטים | ✅ | github, huggingface | github, huggingface-models, huggingface-datasets | שאילתה: llm language model |
| HF04 | מצא מודלים לזיהוי תנוחות גוף | ✅ | github, huggingface | github, huggingface-models, huggingface-datasets | שאילתה: llm language model |
| HF05 | מצא מודלים ל-WebGPU | ✅ | github, huggingface | github, huggingface-models, huggingface-datasets | שאילתה: WebGPU |
| HF06 | מצא מודלים שמתאימים להרצה בדפדפן | ✅ | github, huggingface | github | שאילתה: llm language model |

### stress

| ID | שאלה | סטטוס | Intents | מקורות | הערה |
|----|------|--------|---------|---------|------|
| ST01 | תן לי תמונת מצב מלאה של כדור הארץ כרגע | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (7 מקורות) — דורש LLM לשילוב |
| ST02 | מה 20 האירועים החריגים ביותר שמתרחשים עכשיו? | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (7 מקורות) — דורש LLM לשילוב |
| ST03 | הצג על הגלובוס בו זמנית מטוסים, אוניות, רעידות אדמה, סו | ✅ | ships, earthquake, aviation, disaster | — | globe: focusEarthquakes |
| ST04 | סכם את כל ההתראות הפעילות בעולם | ⚠️ | disaster, earthquake, aviation, ships, news, hackernews, weather, marine, satellite, alerts | usgs-earthquake, ais-ships, news-rss, adsb-aviation, israel-alerts, gdacs-disasters, hacker-news | חיפוש חלקי (7 מקורות) — דורש LLM לשילוב |

## הצעות בממשק

- **83** שאלות בדיקה במערכת
- **58** הצעות ב-`LANDING_CAPABILITY_CHIPS` — **3** מתחלפות כל **10 שניות**

