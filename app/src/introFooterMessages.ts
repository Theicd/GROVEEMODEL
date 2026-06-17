export type IntroFooterMessage = {
  id: string;
  tag: string;
  text: string;
  warn?: boolean;
};

export const INTRO_FOOTER_MESSAGES: IntroFooterMessage[] = [
  { id: "free", tag: "חינמי", text: "צ'אט AI בדפדפן · Gemma 4 E2B · בלי הרשמה" },
  { id: "privacy", tag: "פרטיות", text: "השיחה נשארת במחשב · בלי שליחה לענן" },
  { id: "browser", tag: "דפדפן", text: "Chrome · Edge · Firefox עדכניים · WebGPU מומלץ" },
  { id: "ram", tag: "דרישות", text: "מינימום 8GB RAM · ~3.9GB הורדה בפעם הראשונה" },
  { id: "games", tag: "משחקים", text: "גלישה, המלצות וקטגוריות — ישירות מהדפדפן" },
  { id: "caps", tag: "יכולות", text: "צ'אט · קוד · ראייה · RSS · מזג אוויר · גלובוס" },
  { id: "local", tag: "מקומי", text: "המודל רץ אצלך — לא שרת AI חיצוני לשיחה" },
  { id: "simple", tag: "פשוט", text: "לחץ «טען מודל לדפדפן» — ומתחילים לדבר" },
];

export function footerPerfMessage(webgpu: boolean): IntroFooterMessage {
  return webgpu
    ? { id: "perf", tag: "ביצועים", text: "WebGPU פעיל — מהירות GPU" }
    : { id: "perf", tag: "ביצועים", text: "WebGPU לא זמין — WASM (CPU), איטי יותר", warn: true };
}
