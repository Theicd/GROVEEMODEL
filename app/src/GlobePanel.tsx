import { useCallback, useEffect, useRef, useState } from "react";
import {
  getRealityEmbedSrc,
  sendGlobeCommand,
  sendGlobeUserRegion,
  type GlobeCommand,
  type GlobeLayersState,
} from "./realityGlobe/bridge";
import { fetchUserGeoRegion, type UserGeoRegion } from "./realityGlobe/geoLocation";
import {
  DEFAULT_GLOBE_LAYERS,
  GlobeLayerToggles,
  PRESENTATION_GLOBE_LAYERS,
} from "./realityGlobe/GlobeLayerToggles";
import { GlobeGaugeStrip } from "./realityGlobe/GlobeGaugeStrip";
import { GlobeNewsTicker } from "./realityGlobe/GlobeNewsTicker";
import { GlobeSciFiFlash } from "./realityGlobe/GlobeSciFiFlash";
import { useGlobeAiHeadlines } from "./realityGlobe/useGlobeAiHeadlines";
import { useGlobeIntelFeed, useGlobeLayersFromMessage } from "./realityGlobe/useGlobeIntelFeed";
import type { IntelTickerItem } from "./realityGlobe/intelFeed";

type Props = {
  onClose: () => void;
  command?: GlobeCommand | null;
  onCommandSent?: () => void;
  modelReady?: boolean;
};

function pingIframeResize(iframe: HTMLIFrameElement | null) {
  iframe?.contentWindow?.postMessage({ source: "grovee", type: "resize" }, "*");
}

function isPresentationCommand(cmd: GlobeCommand | null | undefined): boolean {
  if (!cmd) return false;
  if (cmd.type === "focusPlaceQuiet") return cmd.presentation !== false;
  return false;
}

export function GlobePanel({ onClose, command, onCommandSent, modelReady = false }: Props) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false);
  const [iframeLoaded, setIframeLoaded] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [mapFlash, setMapFlash] = useState(false);
  const [presentationMode, setPresentationMode] = useState(false);
  const [layers, setLayers] = useState<GlobeLayersState>(DEFAULT_GLOBE_LAYERS);
  const [userGeo, setUserGeo] = useState<UserGeoRegion | null>(null);
  const pendingRef = useRef<GlobeCommand | null>(null);
  const regionSentRef = useRef(false);
  const skipUserRegionRef = useRef(false);
  const intelActive = !presentationMode;
  const { timeline, gauges, updatedAt, snapshot, activeFlash, dismissFlash } =
    useGlobeIntelFeed(intelActive);
  const aiHeadlines = useGlobeAiHeadlines(intelActive, snapshot, userGeo, modelReady);
  const headlines = aiHeadlines.length > 0 ? aiHeadlines : snapshot.headlines;

  const onLayersFromIframe = useCallback((next: GlobeLayersState) => {
    setLayers((prev) => ({ ...prev, ...next }));
  }, []);
  useGlobeLayersFromMessage(onLayersFromIframe);

  useEffect(() => {
    void fetchUserGeoRegion().then(setUserGeo);
  }, []);

  useEffect(() => {
    if (!command) return;
    const pres = isPresentationCommand(command);
    skipUserRegionRef.current = pres;
    if (pres) {
      setPresentationMode(true);
      setLayers({ ...PRESENTATION_GLOBE_LAYERS });
      dismissFlash();
    }
  }, [command, dismissFlash]);

  const pushUserRegion = useCallback(() => {
    if (!userGeo || regionSentRef.current || skipUserRegionRef.current) return;
    sendGlobeUserRegion(iframeRef.current, {
      countryCode: userGeo.countryCode,
      name: userGeo.countryName,
      lat: userGeo.lat,
      lon: userGeo.lon,
    });
    regionSentRef.current = true;
  }, [userGeo]);

  useEffect(() => {
    if (userGeo && (ready || iframeLoaded) && !skipUserRegionRef.current) pushUserRegion();
  }, [userGeo, ready, iframeLoaded, pushUserRegion]);

  const flushCommand = useCallback(
    (cmd: GlobeCommand) => {
      sendGlobeCommand(iframeRef.current, cmd);
      onCommandSent?.();
    },
    [onCommandSent],
  );

  const initIframeSound = useCallback(() => {
    sendGlobeCommand(iframeRef.current, { type: "initSound" });
  }, []);

  const handleTickerClick = useCallback(
    (item: IntelTickerItem) => {
      initIframeSound();
      const lat = item.lat ?? item.geo?.lat;
      const lon = item.lon ?? item.geo?.lon;
      if (lat != null && lon != null) {
        sendGlobeCommand(iframeRef.current, {
          type: "flyToAlert",
          lat,
          lon,
          severity: item.severity,
          category: item.category || item.tag,
        });
        return;
      }
      if (item.category === "ISRAEL") {
        sendGlobeCommand(iframeRef.current, { type: "focusIsrael" });
      }
    },
    [initIframeSound],
  );

  const handleLayerToggle = useCallback(() => {
    if (presentationMode) {
      setPresentationMode(false);
      skipUserRegionRef.current = false;
      sendGlobeCommand(iframeRef.current, { type: "setPresentationMode", on: false });
    }
  }, [presentationMode]);

  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.data?.source !== "reality-core") return;
      if (e.data.type === "ready") {
        setReady(true);
        setLoadError(false);
        pingIframeResize(iframeRef.current);
        initIframeSound();
        if (!skipUserRegionRef.current) pushUserRegion();
        if (pendingRef.current) {
          flushCommand(pendingRef.current);
          pendingRef.current = null;
        }
      }
      if (e.data.type === "error") setLoadError(true);
      if (e.data.type === "alert_flash" && !presentationMode) {
        setMapFlash(true);
        window.setTimeout(() => setMapFlash(false), 650);
      }
      if (e.data.type === "presentation" && e.data.payload?.on === false) {
        setPresentationMode(false);
      }
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, [flushCommand, pushUserRegion, initIframeSound, presentationMode]);

  const readyRef = useRef(false);
  readyRef.current = ready;

  useEffect(() => {
    const t = window.setTimeout(() => {
      if (!readyRef.current) setLoadError(true);
    }, 30_000);
    return () => window.clearTimeout(t);
  }, []);

  useEffect(() => {
    if (!command) return;
    if (ready) flushCommand(command);
    else pendingRef.current = command;
  }, [command, ready, flushCommand]);

  const handleIframeLoad = useCallback(() => {
    setIframeLoaded(true);
    setLoadError(false);
    const iframe = iframeRef.current;
    pingIframeResize(iframe);
    if (!skipUserRegionRef.current) pushUserRegion();
    window.setTimeout(() => pingIframeResize(iframe), 400);
    window.setTimeout(() => pingIframeResize(iframe), 1200);
  }, [pushUserRegion]);

  const showLoading = !ready && !loadError && !iframeLoaded;

  return (
    <div
      className={`globe-panel globe-panel--embed${presentationMode ? " globe-panel--presentation" : ""}`}
      dir="rtl"
    >
      <header className="globe-embed-bar">
        <div className="globe-embed-brand">
          <span className="globe-live-dot" />
          <span className="globe-embed-title">REALITY LIVE</span>
          {presentationMode ? (
            <span className="globe-embed-region globe-embed-region--quiet" title="מצב תצוגה">
              תצוגה
            </span>
          ) : userGeo ? (
            <span className="globe-embed-region" title={userGeo.countryName}>
              {userGeo.countryCode}
            </span>
          ) : null}
        </div>
        <GlobeLayerToggles
          iframeRef={iframeRef}
          layers={layers}
          onLayersChange={(next) => {
            handleLayerToggle();
            setLayers(next);
          }}
        />
        <button type="button" className="globe-panel-close" onClick={onClose} aria-label="סגור">
          ✕
        </button>
      </header>

      <div
        className={`globe-panel-map${mapFlash ? " globe-panel-map--flash" : ""}`}
        onClick={initIframeSound}
        role="presentation"
      >
        {showLoading && (
          <div className="globe-panel-loading" aria-live="polite">
            <span className="globe-panel-loading-spinner" />
            <p>טוען גלובוס…</p>
          </div>
        )}

        {loadError && !ready && (
          <div className="globe-panel-offline">
            <p>הגלובוס לא נטען.</p>
          </div>
        )}

        {!presentationMode && <GlobeSciFiFlash alert={activeFlash} onDismiss={dismissFlash} />}

        <iframe
          ref={iframeRef}
          className="globe-embed-frame"
          src={getRealityEmbedSrc()}
          title="Reality Core — Cesium Globe"
          allow="fullscreen"
          onLoad={handleIframeLoad}
        />
      </div>

      {!presentationMode && (
        <>
          <GlobeGaugeStrip gauges={gauges} updatedAt={updatedAt} />
          <GlobeNewsTicker
            items={timeline}
            headlines={headlines}
            loading={timeline.length === 0}
            onItemClick={handleTickerClick}
          />
        </>
      )}
    </div>
  );
}
