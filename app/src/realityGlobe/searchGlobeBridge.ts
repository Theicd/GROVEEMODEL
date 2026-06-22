import type { SearchSourceResult } from "../webSearch/types";
import type { GlobeCommand } from "./bridge";
import { isGlobePresentationQuery } from "./intents";

const MAP_REQUEST =
  /על\s+המפה|on\s+the\s+map|תראה\s*(?:לי\s+)?(?:על\s+)?(?:ה)?מפה|show\s+(?:me\s+)?(?:on\s+)?(?:the\s+)?map|הראה\s*(?:לי\s+)?(?:על\s+)?(?:ה)?מפה|הצג\s*(?:לי\s+)?(?:על\s+)?(?:ה)?מפה/i;

const ROUTE_REQUEST =
  /(?:איך\s+מגיע|ניווט|navigation|מסלול|route|דרך\s+ל|how\s+to\s+get)/i;

export function parseCoordsFromNominatimText(text: string): { lat: number; lon: number; label?: string } | null {
  const line = text.split("\n").find((l) => /\d+\.\d+,\s*\d+\.\d+/.test(l));
  if (!line) return null;
  const coords = line.match(/(\d+\.\d+),\s*(\d+\.\d+)/);
  if (!coords) return null;
  const titleLine = text.split("\n").find((l) => /^\d+\./.test(l.trim()));
  const label = titleLine?.replace(/^\d+\.\s*/, "").split("(")[0]?.trim();
  return { lat: parseFloat(coords[1]), lon: parseFloat(coords[2]), label };
}

export function isMapOrRouteRequest(query: string): boolean {
  return MAP_REQUEST.test(query) || ROUTE_REQUEST.test(query);
}

export function shouldOpenGlobeForStructuredGeo(
  query: string,
  intents: string[],
  sources: SearchSourceResult[],
): boolean {
  if (isGlobePresentationQuery(query) || MAP_REQUEST.test(query)) return true;
  if (intents.includes("distance") && ROUTE_REQUEST.test(query)) return true;
  if (intents.includes("places") || intents.includes("distance")) {
    const geoSource = sources.find(
      (s) =>
        (s.provider === "nominatim-places" || s.provider === "osrm-distance") &&
        s.ok &&
        (s.geo?.lat != null || (s.geo?.route?.length ?? 0) > 0 || parseCoordsFromNominatimText(s.text)),
    );
    if (geoSource) return true;
  }
  return false;
}

export function buildGlobeCommandFromSearch(
  query: string,
  intents: string[],
  sources: SearchSourceResult[],
): GlobeCommand | null {
  const presentation = isMapOrRouteRequest(query) || isGlobePresentationQuery(query);

  const dist = sources.find((s) => s.provider === "osrm-distance" && s.ok);
  if (dist?.geo?.route?.length) {
    return {
      type: "drawRoute",
      points: dist.geo.route,
      label: dist.geo.label ?? "מסלול",
      presentation,
    };
  }
  if (dist?.geo?.from && dist?.geo?.to && (presentation || intents.includes("distance"))) {
    return {
      type: "drawRoute",
      points: [dist.geo.from, dist.geo.to],
      label: dist.geo.label ?? "מסלול",
      presentation,
    };
  }

  const places = sources.find((s) => s.provider === "nominatim-places" && s.ok);
  if (places) {
    const geo =
      places.geo?.lat != null && places.geo.lon != null
        ? { lat: places.geo.lat, lon: places.geo.lon, label: places.geo.label }
        : parseCoordsFromNominatimText(places.text);
    if (geo) {
      return {
        type: "flyTo",
        lat: geo.lat,
        lon: geo.lon,
        alt: 8000,
        label: geo.label,
        presentation: presentation || intents.includes("places"),
      };
    }
    const nameLine = places.text.match(/^\d+\.\s+([^\n(]+)/m);
    if (nameLine && presentation) {
      return { type: "focusPlaceQuiet", name: nameLine[1].trim(), presentation: true };
    }
  }

  return null;
}
