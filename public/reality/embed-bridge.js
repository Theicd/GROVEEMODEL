/**
 * GROVEE ↔ Reality Core embed bridge (postMessage).
 */
(function embedBridge() {
  const params = new URLSearchParams(location.search);
  if (params.get('embed') !== 'grovee') return;

  const getViewer = () => {
    if (typeof viewer !== 'undefined' && viewer) return viewer;
    if (typeof cesiumViewer !== 'undefined' && cesiumViewer) return cesiumViewer;
    return null;
  };

  function resizeViewer() {
    const v = getViewer();
    if (!v) return;
    try {
      v.resize();
      v.scene.requestRender();
    } catch (_) {}
  }

  const waitForViewer = (maxMs = 60000) =>
    new Promise((resolve, reject) => {
      const t0 = Date.now();
      const tick = () => {
        const v = getViewer();
        if (v) return resolve(v);
        if (Date.now() - t0 > maxMs) return reject(new Error('Cesium timeout'));
        setTimeout(tick, 150);
      };
      tick();
    });

  async function geocode(name) {
    const q = String(name || '').trim();
    if (!q) return null;
    if (typeof resolvePlaceGeo === 'function') {
      const preset = resolvePlaceGeo(q);
      if (preset) return { lat: preset.lat, lon: preset.lon, label: preset.label || q };
    }
    try {
      const r = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=1&language=he&format=json`,
      );
      const j = await r.json();
      const p = j.results?.[0];
      if (!p) return null;
      return { lat: p.latitude, lon: p.longitude, label: p.name };
    } catch {
      return null;
    }
  }

  async function flyToCoords(lon, lat, alt = 800000, quiet = false) {
    try {
      await waitForViewer();
      resizeViewer();
      if (quiet && typeof focusPlaceQuiet === 'function') {
        focusPlaceQuiet(lon, lat, alt);
        return;
      }
      if (typeof focusPlaceQuiet === 'function') {
        focusPlaceQuiet(lon, lat, alt);
        return;
      }
      const v = getViewer();
      v.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
        duration: 2.2,
        easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
      });
    } catch (e) {
      console.warn('[embed-bridge] flyTo failed', e);
    }
  }

  async function focusPlaceQuietCommand(name, alt, presentation) {
    if (presentation && typeof setPresentationMode === 'function') setPresentationMode(true);
    if (typeof focusPlaceByName === 'function') {
      const hit = await focusPlaceByName(name, alt);
      if (hit) return hit;
    }
    const g = await geocode(name);
    if (g) {
      if (typeof focusPlaceQuiet === 'function') {
        focusPlaceQuiet(g.lon, g.lat, alt || 900000, g.label || name);
      } else {
        await flyToCoords(g.lon, g.lat, alt || 900000, true);
      }
      return g;
    }
    return null;
  }

  function getEarthquakes() {
    if (typeof live !== 'undefined' && live.earthquake?.items) return live.earthquake.items;
    if (typeof data !== 'undefined' && data.earthquake?.items) return data.earthquake.items;
    return [];
  }

  async function handleCommand(type, payload = {}) {
    const p = payload.payload || payload;

    if (type === 'toggleLayer') {
      const layer = p.layer;
      if (typeof toggleEmbedLayer === 'function') {
        const layers = toggleEmbedLayer(layer);
        window.parent.postMessage({ source: 'reality-core', type: 'layers', payload: layers }, '*');
      }
      return;
    }
    if (type === 'setUserRegion') {
      if (typeof setEmbedUserRegion === 'function') setEmbedUserRegion(p);
      return;
    }
    if (type === 'resize') {
      resizeViewer();
      return;
    }
    if (type === 'initSound') {
      window.soundSystem?.init();
      return;
    }
    if (type === 'playSound') {
      if (p.kind === 'critical') window.soundSystem?.alertCritical();
      else window.soundSystem?.alertInfo();
      return;
    }
    if (type === 'setPresentationMode') {
      if (typeof setPresentationMode === 'function') setPresentationMode(!!p.on);
      return;
    }
    if (type === 'setMapMode' && p.mode) {
      if (typeof setMapMode === 'function') setMapMode(p.mode, p.fly !== false);
      resizeViewer();
      return;
    }
    if (type === 'focusPlaceQuiet' && p.name) {
      await focusPlaceQuietCommand(p.name, p.alt, p.presentation !== false);
      return;
    }
    if (type === 'flyToAlert' && p.lat != null && p.lon != null) {
      if (typeof flyToAlertAndReturn === 'function') {
        flyToAlertAndReturn({
          geo: { lat: p.lat, lon: p.lon },
          severity: p.severity || 4,
          category: p.category || 'ALERT',
        });
      } else {
        await flyToCoords(p.lon, p.lat, p.alt || 800000, true);
      }
      return;
    }
    if (type === 'flyTo' && p.lat != null && p.lon != null) {
      await flyToCoords(p.lon, p.lat, p.alt || 800000, true);
      return;
    }
    if (type === 'focusPlace' && p.name) {
      await focusPlaceQuietCommand(p.name, p.alt, true);
      return;
    }
    if (type === 'showLayer') {
      if (typeof setPresentationMode === 'function') setPresentationMode(false);
      if (typeof setMapMode === 'function') setMapMode('israel_flat', true);
      resizeViewer();
      return;
    }
    if (type === 'focusEarthquakes') {
      if (typeof setPresentationMode === 'function') setPresentationMode(false);
      if (typeof setMapMode === 'function') setMapMode('israel_flat', false);
      const eq = getEarthquakes()[0];
      if (eq?.geo && typeof flyToAlertAndReturn === 'function') {
        flyToAlertAndReturn({ geo: eq.geo, severity: 4, category: 'SEISMIC' });
      } else if (eq?.geo) await flyToCoords(eq.geo.lon, eq.geo.lat, 600000);
      return;
    }
    if (type === 'focusIsrael') {
      if (typeof focusEmbedHomeView === 'function') focusEmbedHomeView(1.4);
      else if (typeof focusIsrael === 'function') focusIsrael(1.4);
      resizeViewer();
      return;
    }
    if (type === 'globe3d') {
      if (typeof setMapMode === 'function') setMapMode('globe', true);
      resizeViewer();
    }
  }

  window.addEventListener('message', (e) => {
    if (!e.data || e.data.source !== 'grovee') return;
    void handleCommand(e.data.type, e.data.payload || e.data);
  });

  window.addEventListener('resize', () => resizeViewer());

  waitForViewer()
    .then(() => {
      resizeViewer();
      setTimeout(resizeViewer, 300);
      setTimeout(resizeViewer, 1000);
      window.parent.postMessage({ source: 'reality-core', type: 'ready' }, '*');
    })
    .catch(() => {
      window.parent.postMessage(
        { source: 'reality-core', type: 'error', payload: { message: 'Cesium failed' } },
        '*',
      );
    });

  window.realityCoreEmbed = { flyToCoords, geocode, handleCommand, resizeViewer, focusPlaceQuietCommand };
})();
