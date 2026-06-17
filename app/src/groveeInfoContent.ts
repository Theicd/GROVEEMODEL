export type GroveeInfoCard = {
  id: string;
  icon: string;
  title: string;
  body: string;
  tags?: readonly string[];
  links?: readonly { label: string; href: string }[];
};

/** Compact copy — fits one full-screen grid without scrolling. */
export const GROVEE_INFO_CARDS: readonly GroveeInfoCard[] = [
  {
    id: "engine",
    icon: "⚡",
    title: "מנוע מקומי",
    body: "AI בדפדפן — WebAssembly ו-Transformers.js. בלי התקנה, בלי שרת AI לשיחה.",
    tags: ["ONNX", "Transformers.js", "WASM"],
  },
  {
    id: "privacy",
    icon: "🔒",
    title: "פרטיות",
    body: "מה שאתה כותב נשאר במכשיר. חיפוש ברשת רק לשאלות על מידע עדכני.",
    tags: ["מקומי", "ללא ענן AI"],
  },
  {
    id: "model",
    icon: "🧠",
    title: "GEMMA 4 E2B",
    body: "~3.9GB בטעינה ראשונה — שיחה, ראייה ו-Think. נשמר במטמון הדפדפן.",
    tags: ["~3.9GB", "WebGPU / WASM"],
  },
  {
    id: "capabilities",
    icon: "✦",
    title: "יכולות",
    body: "צ'אט · קוד · HTML · תמונות · מצלמה · RSS · מזג אוויר · גלובוס · משחקים.",
    tags: ["צ'אט", "ראייה", "Globe"],
  },
  {
    id: "sources",
    icon: "🌐",
    title: "מידע חי",
    body: "שאילתות עדכניות — ממקורות ציבוריים, לא מהמודל בלבד.",
    links: [
      { label: "Time.Now", href: "https://time.now" },
      { label: "Open-Meteo", href: "https://open-meteo.com" },
      { label: "OSM", href: "https://www.openstreetmap.org" },
    ],
  },
] as const;
