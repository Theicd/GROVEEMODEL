import type { ReactElement, ReactNode } from "react";

import type { GroveeInfoCard } from "../groveeInfoContent";

const accent = "#10a37f";
const cyan = "#00f3ff";
const dim = "#5a6a7a";
const line = "rgba(0, 243, 255, 0.35)";

function IllustrationFrame({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div className="grovee-info-card__visual" aria-hidden="true">
      <svg viewBox="0 0 320 140" className="grovee-info-card__svg" preserveAspectRatio="xMidYMid meet">
        {children}
      </svg>
      <span className="grovee-info-card__visual-label" dir="ltr">
        {label}
      </span>
    </div>
  );
}

function EngineIllustration() {
  return (
    <IllustrationFrame label="Browser → WASM → ONNX → Gemma">
      <rect x="8" y="12" width="304" height="116" rx="8" fill="#0a1018" stroke={line} strokeWidth="1.2" />
      <rect x="8" y="12" width="304" height="22" rx="8" fill="#121a24" />
      <circle cx="22" cy="23" r="3" fill="#ff5f57" />
      <circle cx="34" cy="23" r="3" fill="#febc2e" />
      <circle cx="46" cy="23" r="3" fill="#28c840" />
      <text x="160" y="27" textAnchor="middle" fill={dim} fontSize="8" fontFamily="monospace">
        Chrome · GROVEE
      </text>
      <rect x="24" y="44" width="72" height="68" rx="6" fill="rgba(16,163,127,0.12)" stroke={accent} strokeWidth="1" />
      <text x="60" y="62" textAnchor="middle" fill={accent} fontSize="9" fontWeight="600">
        WASM
      </text>
      <text x="60" y="78" textAnchor="middle" fill={dim} fontSize="7">
        CPU/GPU
      </text>
      <path d="M96 78h24" stroke={cyan} strokeWidth="1.5" />
      <rect x="120" y="44" width="72" height="68" rx="6" fill="rgba(0,243,255,0.08)" stroke={cyan} strokeWidth="1" />
      <text x="156" y="62" textAnchor="middle" fill={cyan} fontSize="9" fontWeight="600">
        ONNX
      </text>
      <text x="156" y="78" textAnchor="middle" fill={dim} fontSize="7">
        Runtime
      </text>
      <path d="M192 78h24" stroke={cyan} strokeWidth="1.5" />
      <polygon points="216,78 224,74 224,82" fill={cyan} />
      <rect x="224" y="44" width="72" height="68" rx="6" fill="rgba(16,163,127,0.18)" stroke={accent} strokeWidth="1.2" />
      <text x="260" y="62" textAnchor="middle" fill="#fff" fontSize="9" fontWeight="600">
        Gemma
      </text>
      <text x="260" y="78" textAnchor="middle" fill={dim} fontSize="7">
        4 E2B
      </text>
      <text x="160" y="128" textAnchor="middle" fill={dim} fontSize="8">
        הכול רץ אצלך — בלי שרת AI
      </text>
    </IllustrationFrame>
  );
}

function PrivacyIllustration() {
  return (
    <IllustrationFrame label="Local only · no AI cloud">
      <rect x="30" y="20" width="120" height="90" rx="10" fill="#0a1018" stroke={accent} strokeWidth="1.5" />
      <rect x="42" y="32" width="96" height="54" rx="4" fill="#060a10" stroke={line} />
      <circle cx="90" cy="52" r="14" fill="none" stroke={accent} strokeWidth="2" />
      <rect x="84" y="58" width="12" height="14" rx="2" fill={accent} />
      <text x="90" y="98" textAnchor="middle" fill={accent} fontSize="9" fontWeight="600">
        המכשיר שלך
      </text>
      <path d="M162 55h40" stroke="#ff6b6b" strokeWidth="1.5" strokeDasharray="4 3" />
      <text x="182" y="48" textAnchor="middle" fill="#ff6b6b" fontSize="14">
        ✕
      </text>
      <rect x="210" y="28" width="80" height="74" rx="8" fill="#0f0808" stroke="rgba(255,80,80,0.35)" strokeWidth="1" strokeDasharray="5 4" />
      <text x="250" y="52" textAnchor="middle" fill="#888" fontSize="8">
        ענן AI
      </text>
      <text x="250" y="68" textAnchor="middle" fill="#666" fontSize="7">
        (לא נשלח)
      </text>
      <text x="160" y="128" textAnchor="middle" fill={dim} fontSize="8">
        שיחה = מקומית · חיפוש = מקורות ציבוריים בלבד
      </text>
    </IllustrationFrame>
  );
}

function ModelIllustration() {
  return (
    <IllustrationFrame label="~3.9GB · browser cache">
      <circle cx="100" cy="72" r="48" fill="none" stroke="#1a2530" strokeWidth="8" />
      <circle
        cx="100"
        cy="72"
        r="48"
        fill="none"
        stroke={cyan}
        strokeWidth="8"
        strokeDasharray="180 120"
        strokeLinecap="round"
        transform="rotate(-90 100 72)"
      />
      <text x="100" y="68" textAnchor="middle" fill="#fff" fontSize="14" fontWeight="700">
        72%
      </text>
      <text x="100" y="82" textAnchor="middle" fill={dim} fontSize="7">
        loading
      </text>
      <path d="M158 72h24" stroke={cyan} strokeWidth="1.5" />
      <polygon points="182,72 190,68 190,76" fill={cyan} />
      <rect x="198" y="36" width="100" height="72" rx="8" fill="#0a1018" stroke={line} strokeWidth="1" />
      <text x="248" y="58" textAnchor="middle" fill={cyan} fontSize="8" fontFamily="monospace">
        Cache
      </text>
      <rect x="210" y="66" width="76" height="8" rx="2" fill="#1a2530" />
      <rect x="210" y="66" width="55" height="8" rx="2" fill={accent} />
      <rect x="210" y="80" width="76" height="8" rx="2" fill="#1a2530" />
      <rect x="210" y="80" width="62" height="8" rx="2" fill={cyan} opacity="0.7" />
      <text x="248" y="102" textAnchor="middle" fill={dim} fontSize="7">
        vision + text
      </text>
      <text x="160" y="128" textAnchor="middle" fill={dim} fontSize="8">
        פעם ראשונה ארוכה · אחר כך מהיר מהמטמון
      </text>
    </IllustrationFrame>
  );
}

function CapabilitiesIllustration() {
  const items = [
    { x: 40, label: "צ'אט", icon: "💬" },
    { x: 100, label: "קוד", icon: "</>" },
    { x: 160, label: "תמונה", icon: "🖼" },
    { x: 220, label: "מצלמה", icon: "📷" },
    { x: 280, label: "Globe", icon: "🌍" },
  ];
  const row2 = [
    { x: 70, label: "RSS", icon: "📰" },
    { x: 130, label: "מזג", icon: "☁" },
    { x: 190, label: "HTML", icon: "◇" },
    { x: 250, label: "משחק", icon: "🎮" },
  ];
  return (
    <IllustrationFrame label="All inside one interface">
      {items.map((it) => (
        <g key={it.label}>
          <rect x={it.x - 24} y="28" width="48" height="48" rx="8" fill="rgba(0,243,255,0.06)" stroke={line} />
          <text x={it.x} y="50" textAnchor="middle" fontSize="14">
            {it.icon}
          </text>
          <text x={it.x} y="88" textAnchor="middle" fill={dim} fontSize="7">
            {it.label}
          </text>
        </g>
      ))}
      {row2.map((it) => (
        <g key={it.label}>
          <rect x={it.x - 22} y="96" width="44" height="22" rx="6" fill="rgba(16,163,127,0.1)" stroke={accent} strokeWidth="0.8" />
          <text x={it.x} y="106" textAnchor="middle" fill="#ccc" fontSize="8">
            {it.icon} {it.label}
          </text>
        </g>
      ))}
    </IllustrationFrame>
  );
}

function SourcesIllustration() {
  return (
    <IllustrationFrame label="Live feeds → your browser">
      <circle cx="160" cy="68" r="36" fill="rgba(0,243,255,0.06)" stroke={cyan} strokeWidth="1.2" />
      <ellipse cx="160" cy="68" rx="36" ry="14" fill="none" stroke={line} strokeWidth="0.8" />
      <ellipse cx="160" cy="68" rx="14" ry="36" fill="none" stroke={line} strokeWidth="0.8" />
      <line x1="124" y1="68" x2="196" y2="68" stroke={line} strokeWidth="0.8" />
      <circle cx="48" cy="40" r="16" fill="#0a1018" stroke={accent} strokeWidth="1" />
      <text x="48" y="44" textAnchor="middle" fill={accent} fontSize="8">
        RSS
      </text>
      <path d="M64 44 L128 58" stroke={accent} strokeWidth="1" opacity="0.7" />
      <circle cx="48" cy="96" r="16" fill="#0a1018" stroke={cyan} strokeWidth="1" />
      <text x="48" y="100" textAnchor="middle" fill={cyan} fontSize="7">
        Time
      </text>
      <path d="M64 92 L128 74" stroke={cyan} strokeWidth="1" opacity="0.7" />
      <circle cx="272" cy="40" r="16" fill="#0a1018" stroke={cyan} strokeWidth="1" />
      <text x="272" y="44" textAnchor="middle" fill={cyan} fontSize="7">
        Meteo
      </text>
      <path d="M256 44 L192 58" stroke={cyan} strokeWidth="1" opacity="0.7" />
      <circle cx="272" cy="96" r="16" fill="#0a1018" stroke={accent} strokeWidth="1" />
      <text x="272" y="100" textAnchor="middle" fill={accent} fontSize="7">
        OSM
      </text>
      <path d="M256 92 L192 74" stroke={accent} strokeWidth="1" opacity="0.7" />
      <text x="160" y="128" textAnchor="middle" fill={dim} fontSize="8">
        עובדות עדכניות — לא מהמודל בלבד
      </text>
    </IllustrationFrame>
  );
}

const ILLUSTRATIONS: Record<GroveeInfoCard["id"], () => ReactElement> = {
  engine: EngineIllustration,
  privacy: PrivacyIllustration,
  model: ModelIllustration,
  capabilities: CapabilitiesIllustration,
  sources: SourcesIllustration,
};

export function GroveeInfoIllustration({ cardId }: { cardId: GroveeInfoCard["id"] }) {
  const Component = ILLUSTRATIONS[cardId];
  return <Component />;
}
