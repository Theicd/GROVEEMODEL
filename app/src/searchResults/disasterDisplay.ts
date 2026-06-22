export type DisasterTypeCode = "EQ" | "TC" | "FL" | "VO" | "WF" | "DR" | "UNKNOWN";

export type AlertSeverity = "red" | "orange" | "green" | "unknown";

export type DisasterTypeInfo = {
  code: DisasterTypeCode;
  icon: string;
  labelHe: string;
  labelEn: string;
  css: string;
};

export type AlertLevelInfo = {
  severity: AlertSeverity;
  labelHe: string;
  labelEn: string;
  css: string;
  hintHe: string;
  hintEn: string;
};

const TYPE_BY_CODE: Record<DisasterTypeCode, Omit<DisasterTypeInfo, "code">> = {
  EQ: { icon: "🫨", labelHe: "רעידת אדמה", labelEn: "Earthquake", css: "serp-disaster-type--eq" },
  TC: { icon: "🌀", labelHe: "הוריקן / סופה", labelEn: "Cyclone / Hurricane", css: "serp-disaster-type--tc" },
  FL: { icon: "🌊", labelHe: "הצפה", labelEn: "Flood", css: "serp-disaster-type--fl" },
  VO: { icon: "🌋", labelHe: "התפרצות הר געש", labelEn: "Volcano", css: "serp-disaster-type--vo" },
  WF: { icon: "🔥", labelHe: "שריפה", labelEn: "Wildfire", css: "serp-disaster-type--wf" },
  DR: { icon: "🏜️", labelHe: "בצורת", labelEn: "Drought", css: "serp-disaster-type--dr" },
  UNKNOWN: { icon: "⚠️", labelHe: "אסון טבע", labelEn: "Natural disaster", css: "serp-disaster-type--unknown" },
};

const inferTypeFromText = (text: string): DisasterTypeCode => {
  const t = text.toLowerCase();
  if (/\b(?:eq|earthquake|seismic|רעיד)/i.test(t)) return "EQ";
  if (/\b(?:tc|cyclone|hurricane|typhoon|tropical|הוריקן|סופה\s*טרופ|ציקלון)/i.test(t)) return "TC";
  if (/\b(?:fl|flood|inundation|הצפה|שיטפון)/i.test(t)) return "FL";
  if (/\b(?:vo|volcano|volcanic|הר\s*געש|געש)/i.test(t)) return "VO";
  if (/\b(?:wf|wildfire|fire|שריפ|firestorm)/i.test(t)) return "WF";
  if (/\b(?:dr|drought|בצורת)/i.test(t)) return "DR";
  return "UNKNOWN";
};

export const resolveDisasterType = (
  rawType?: string,
  eventName?: string,
): DisasterTypeInfo => {
  const codeRaw = (rawType ?? "").trim().toUpperCase();
  let code: DisasterTypeCode = "UNKNOWN";
  if (codeRaw && codeRaw in TYPE_BY_CODE) {
    code = codeRaw as DisasterTypeCode;
  } else if (codeRaw.length >= 2) {
    code = inferTypeFromText(codeRaw);
  }
  if (code === "UNKNOWN" && eventName) {
    code = inferTypeFromText(eventName);
  }
  return { code, ...TYPE_BY_CODE[code] };
};

export const resolveAlertLevel = (raw?: string): AlertLevelInfo => {
  const level = (raw ?? "").trim();
  if (/red|אדום|חמור/i.test(level)) {
    return {
      severity: "red",
      labelHe: "חמור",
      labelEn: "Severe",
      css: "serp-disaster-severity--red",
      hintHe: "התרעה אדומה — סיכון גבוה, דורש תשומת לב מיידית",
      hintEn: "Red alert — high impact, immediate attention",
    };
  }
  if (/orange|כתום|בינונ/i.test(level)) {
    return {
      severity: "orange",
      labelHe: "בינוני",
      labelEn: "Moderate",
      css: "serp-disaster-severity--orange",
      hintHe: "התרעה כתומה — מצב מעקב פעיל",
      hintEn: "Orange alert — active monitoring",
    };
  }
  if (/green|ירוק|קל/i.test(level)) {
    return {
      severity: "green",
      labelHe: "קל",
      labelEn: "Low",
      css: "serp-disaster-severity--green",
      hintHe: "התרעה ירוקה — סיכון נמוך יחסית",
      hintEn: "Green alert — relatively low risk",
    };
  }
  return {
    severity: "unknown",
    labelHe: "לא ידוע",
    labelEn: "Unknown",
    css: "serp-disaster-severity--unknown",
    hintHe: "רמת התרעה לא זמינה",
    hintEn: "Alert level unavailable",
  };
};
