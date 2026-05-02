/**
 * Lightweight checks before loading Transformers.js / WebGPU stacks.
 * Does not download models — only probes browser + GPU availability.
 */

export type DeviceCheckItem = {
  id: string;
  ok: boolean;
  label: string;
  hint: string;
};

export type DeviceTier = "likely_ok" | "mixed" | "unlikely";

export type DeviceCheckResult = {
  tier: DeviceTier;
  summaryHe: string;
  items: DeviceCheckItem[];
};

function tierLabel(tier: DeviceTier): string {
  switch (tier) {
    case "likely_ok":
      return "מתאים ברוב הסיכויים";
    case "mixed":
      return "יתכנו האטות או צפיפות זיכרון";
    case "unlikely":
      return "קשה יותר — נסו דפדפן/מחשב אחר";
    default:
      return "";
  }
}

export async function runDeviceCapabilityCheck(): Promise<DeviceCheckResult> {
  const items: DeviceCheckItem[] = [];

  const secure = typeof window !== "undefined" && window.isSecureContext;
  items.push({
    id: "secure",
    ok: secure,
    label: "HTTPS / localhost",
    hint: secure
      ? "הקשר מאובטח — נדרש לרוב יכולות ה-ML בדפדפן."
      : "חסר הקשר מאובטח — WebGPU ומטמון מודלים עלולים להיחסם.",
  });

  let webgpuOk = false;
  if (!navigator.gpu) {
    items.push({
      id: "webgpu",
      ok: false,
      label: "WebGPU",
      hint: "הדפדפן לא תומך ב-WebGPU. נסו Chrome או Edge עדכניים.",
    });
  } else {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        items.push({
          id: "webgpu",
          ok: false,
          label: "WebGPU",
          hint: "לא נמצא מתאם GPU. בדקו מנהלי תצוגה או הרצה על סוללה/מצב חיסכון.",
        });
      } else {
        webgpuOk = true;
        const lim = adapter.limits;
        const mb = Math.round(lim.maxBufferSize / (1024 * 1024));
        items.push({
          id: "webgpu",
          ok: true,
          label: "WebGPU",
          hint: `נמצא מתאם GPU (למשל maxBufferSize ~${mb}MB).`,
        });
      }
    } catch {
      items.push({
        id: "webgpu",
        ok: false,
        label: "WebGPU",
        hint: "שגיאה בבדיקת WebGPU.",
      });
    }
  }

  const nav = navigator as Navigator & { deviceMemory?: number };
  if (typeof nav.deviceMemory === "number") {
    const gb = nav.deviceMemory;
    const ok = gb >= 8;
    items.push({
      id: "ram",
      ok,
      label: "זיכרון RAM (אומדן מהדפדפן)",
      hint: ok
        ? `הדפדפן מדווח על ~${gb}GB — בדרך כלל מספיק ל-Gemma+Coder+Vision יחד.`
        : `~${gb}GB — עלול להידחק אחרי טעינת כמה מודלים; מומלץ 8GB+ זמינים לדפדפן.`,
    });
  } else {
    items.push({
      id: "ram",
      ok: true,
      label: "זיכרון RAM",
      hint: "לא ניתן לאמוד כאן. מומלץ לפחות 8GB מערכת; סגרו לשוניות אחרות.",
    });
  }

  const cores = navigator.hardwareConcurrency ?? 0;
  items.push({
    id: "cpu",
    ok: cores >= 4,
    label: "ליבות מעבד (לוגי)",
    hint: cores > 0 ? `${cores} ליבות — WASM וטעינה מהירים יותר עם יותר ליבות.` : "לא דווח — לא קריטי אם WebGPU עובד.",
  });

  const coi = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  items.push({
    id: "coi",
    ok: coi,
    label: "בידוד מקור (COOP/COEP)",
    hint: coi
      ? "מופעל — עשוי לזרז WASM מרובה-תהליכים."
      : "לא מופעל — עדיין עובד; לפעמים ONNX/WASM איטיים יותר.",
  });

  const softFails = items.filter((i) => !i.ok && i.id !== "coi").length;
  let tier: DeviceTier;
  if (!secure || !webgpuOk) {
    tier = "unlikely";
  } else if (softFails >= 2) {
    tier = "mixed";
  } else {
    tier = "likely_ok";
  }

  const summaryHe = `${tierLabel(tier)}. ${!webgpuOk ? "בלי WebGPU עדיין אפשר לנסות WASM, אבל זה יהיה כבד." : "לאחר הטעינה הראשונה המודלים נשמרים במטמון הדפדפן."}`;

  return { tier, summaryHe, items };
}
