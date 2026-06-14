import type { RefObject } from "react";
import { sendGlobeLayerToggle, type GlobeLayersState, type GlobeMapLayer } from "./bridge";

type LayerDef = {
  id: GlobeMapLayer;
  icon: string;
  label: string;
};

const LAYERS: LayerDef[] = [
  { id: "aviation", icon: "✈", label: "מטוסים" },
  { id: "ships", icon: "⛴", label: "ספינות" },
  { id: "marine_infra", icon: "⚓", label: "נכסי ים" },
  { id: "satellite", icon: "🛰", label: "לוויינים" },
  { id: "earthquake", icon: "🌍", label: "רעידות" },
  { id: "weather", icon: "🌤", label: "מזג" },
  { id: "marine", icon: "🌊", label: "ים" },
  { id: "israel_alerts", icon: "🚨", label: "צבע אדום" },
];

type Props = {
  iframeRef: RefObject<HTMLIFrameElement | null>;
  layers: GlobeLayersState;
  onLayersChange: (next: GlobeLayersState) => void;
};

export function GlobeLayerToggles({ iframeRef, layers, onLayersChange }: Props) {
  const toggle = (id: GlobeMapLayer) => {
    const next = { ...layers, [id]: !layers[id] };
    onLayersChange(next);
    sendGlobeLayerToggle(iframeRef.current, id);
  };

  return (
    <div className="globe-layer-toggles" role="toolbar" aria-label="שכבות מפה">
      {LAYERS.map((layer) => {
        const on = layers[layer.id] !== false;
        return (
          <button
            key={layer.id}
            type="button"
            className={`globe-layer-btn${on ? " globe-layer-btn--on" : ""}`}
            onClick={() => toggle(layer.id)}
            aria-pressed={on}
            aria-label={`${layer.label}${on ? " · פעיל" : " · כבוי"}`}
            title={`${on ? "כבה" : "הדלק"} · ${layer.label}`}
          >
            <span className="globe-layer-btn-icon" aria-hidden="true">
              {layer.icon}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export const DEFAULT_GLOBE_LAYERS: GlobeLayersState = {
  aviation: true,
  ships: true,
  marine_infra: true,
  satellite: true,
  earthquake: true,
  weather: true,
  marine: true,
  israel_alerts: true,
};

export const PRESENTATION_GLOBE_LAYERS: GlobeLayersState = {
  aviation: false,
  ships: false,
  marine_infra: false,
  satellite: false,
  earthquake: false,
  weather: false,
  marine: false,
  israel_alerts: false,
};
