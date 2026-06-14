/** Significant alert policy — fly + flash only for disasters / M≥4.5 / Israel alerts. */
(function () {
  'use strict';

  const FLY_DEBOUNCE_MS = 45000;
  const ISRAEL_CATS = new Set(['ISRAEL', 'CORRELATION']);

  let _lastFlyAt = 0;
  let _quietAlerts = false;

  function parseMag(a) {
    const m = Number(a?.magnitude ?? a?.mag);
    if (Number.isFinite(m) && m > 0) return m;
    const s = String(a?.summary || '');
    const hit = s.match(/\bM\s*([\d.]+)/i) || s.match(/([\d.]+)\s*ר(?:י)?כ?t/i);
    return hit ? parseFloat(hit[1]) : 0;
  }

  function parsePlace(a) {
    if (a?.place) return String(a.place);
    const s = String(a?.summary || '');
    const after = s.match(/M[\d.]+\s*(?:·|-)?\s*(.+)$/);
    return after ? after[1].trim() : '';
  }

  function categorySource(cat) {
    const c = String(cat || '').toUpperCase();
    if (c === 'SEISMIC' || c === 'TSUNAMI') return 'USGS / EMSC';
    if (c === 'DISASTER') return 'GDACS / NASA EONET';
    if (c === 'FIRE') return 'NASA FIRMS';
    if (c === 'WEATHER') return 'NWS';
    if (c === 'ISRAEL') return 'פיקוד העורף / צבע אדום';
    return 'REALITY LIVE';
  }

  function alertTitle(a) {
    const cat = String(a?.category || 'ALERT').toUpperCase();
    const mag = parseMag(a);
    if (cat === 'SEISMIC' || cat === 'TSUNAMI') {
      return cat === 'TSUNAMI' ? `צונמי · M${mag.toFixed(1)}` : `רעידת אדמה M${mag.toFixed(1)}`;
    }
    if (cat === 'ISRAEL') return 'צבע אדום';
    if (cat === 'FIRE') return 'שריפה / כיבוש אש';
    if (cat === 'DISASTER') return String(a?.summary || 'אירוע גלובלי').split('·')[0].trim();
    return cat;
  }

  function isSignificantFly(a) {
    const cat = String(a?.category || '').toUpperCase();
    const sev = Number(a?.severity || 0);
    if (ISRAEL_CATS.has(cat)) return sev >= 3;
    if (cat === 'SEISMIC' || cat === 'TSUNAMI') return parseMag(a) >= 4.5;
    if (cat === 'DISASTER' || cat === 'FIRE') return sev >= 4;
    if (cat === 'WEATHER') return sev >= 4;
    return false;
  }

  function canFlyNow(a) {
    if (_quietAlerts) return false;
    const cat = String(a?.category || '').toUpperCase();
    if (ISRAEL_CATS.has(cat)) return true;
    return Date.now() - _lastFlyAt >= FLY_DEBOUNCE_MS;
  }

  function markFlew() {
    _lastFlyAt = Date.now();
  }

  function setQuietAlerts(on) {
    _quietAlerts = !!on;
  }

  function isQuietAlerts() {
    return _quietAlerts;
  }

  function buildLiveAlertPayload(a) {
    const geo = a?.geo;
    const mag = parseMag(a);
    const ts = a?.timestamp || a?.time;
    return {
      id: String(a.id),
      severity: Number(a.severity || 4),
      title: alertTitle(a),
      body: String(a.summary || a.recommended_action || ''),
      category: String(a.category || ''),
      ts: ts ? new Date(ts).getTime() : Date.now(),
      lat: geo?.lat,
      lon: geo?.lon,
      place: parsePlace(a) || (geo?.lat != null ? `${geo.lat.toFixed(2)}°, ${geo.lon.toFixed(2)}°` : ''),
      magnitude: mag > 0 ? mag : undefined,
      depth: a?.depth != null ? Number(a.depth) : undefined,
      source: a?.source || categorySource(a?.category),
      recommendedAction: String(a.recommended_action || ''),
      eventTime: ts ? new Date(ts).toISOString() : undefined,
    };
  }

  window.SignificantAlerts = {
    parseMag,
    parsePlace,
    alertTitle,
    isSignificantFly,
    canFlyNow,
    markFlew,
    setQuietAlerts,
    isQuietAlerts,
    buildLiveAlertPayload,
    categorySource,
    FLY_DEBOUNCE_MS,
  };
})();
