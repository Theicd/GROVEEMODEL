/** Route-marker seeds — shared by chat ships provider and Reality Live globe. */

export type MedPortSeed = {
  name: string;
  lat: number;
  lon: number;
  h: number;
  t: number;
  dst: string;
};

export const MED_PORTS: MedPortSeed[] = [
  { name: "Haifa Cargo", lat: 32.82, lon: 35.0, h: 270, t: 70, dst: "ILHFA" },
  { name: "Haifa Port Route", lat: 32.79, lon: 35.02, h: 270, t: 70, dst: "ILHFA" },
  { name: "Ashdod Container", lat: 31.83, lon: 34.63, h: 250, t: 70, dst: "ILASH" },
  { name: "Eilat Tanker", lat: 29.55, lon: 34.96, h: 180, t: 80, dst: "ILEIL" },
  { name: "Suez Transit", lat: 31.25, lon: 32.31, h: 45, t: 80, dst: "EGPSD" },
  { name: "Med Bulker", lat: 33.2, lon: 33.8, h: 210, t: 70, dst: "CYLMS" },
  { name: "Limassol Ferry", lat: 34.65, lon: 33.03, h: 90, t: 60, dst: "CYLMS" },
  { name: "Piraeus Express", lat: 35.5, lon: 31.0, h: 300, t: 60, dst: "GRPIR" },
  { name: "Med Tanker", lat: 32.5, lon: 33.0, h: 180, t: 80, dst: "EGALY" },
  { name: "Aqaba Star", lat: 29.48, lon: 34.98, h: 0, t: 70, dst: "JOAQJ" },
  { name: "Alexandria Cargo", lat: 31.2, lon: 29.9, h: 90, t: 70, dst: "EGALY" },
  { name: "Mersin Link", lat: 36.6, lon: 34.6, h: 180, t: 60, dst: "TRMER" },
  { name: "Tartus Freight", lat: 34.9, lon: 35.85, h: 270, t: 70, dst: "SYTAR" },
  { name: "Crete Passage", lat: 35.2, lon: 24.5, h: 90, t: 70, dst: "GRHER" },
  { name: "Red Sea Tanker", lat: 27.8, lon: 34.3, h: 180, t: 80, dst: "SAJED" },
  { name: "Bab el-Mandeb", lat: 12.6, lon: 43.3, h: 0, t: 80, dst: "DJJIB" },
  { name: "Gulf Carrier", lat: 26.5, lon: 52.0, h: 90, t: 80, dst: "AEJEA" },
  { name: "Istanbul Ferry", lat: 41.0, lon: 29.0, h: 200, t: 60, dst: "TRIST" },
  { name: "Suez South", lat: 30.0, lon: 32.58, h: 180, t: 70, dst: "EGPSD" },
];

export const WORLD_PORTS: MedPortSeed[] = [
  { name: "NY Container", lat: 40.68, lon: -74.05, h: 270, t: 70, dst: "USNYC" },
  { name: "Rotterdam", lat: 51.92, lon: 4.48, h: 90, t: 70, dst: "NLRTM" },
  { name: "Singapore Strait", lat: 1.25, lon: 103.85, h: 45, t: 80, dst: "SGSIN" },
  { name: "Shanghai", lat: 31.23, lon: 121.47, h: 180, t: 70, dst: "CNSHA" },
  { name: "Tokyo Bay", lat: 35.45, lon: 139.77, h: 90, t: 60, dst: "JPTYO" },
  { name: "LA Long Beach", lat: 33.75, lon: -118.2, h: 270, t: 70, dst: "USLAX" },
  { name: "Panama Transit", lat: 9.08, lon: -79.68, h: 90, t: 80, dst: "PAPTY" },
  { name: "Cape Town", lat: -33.92, lon: 18.42, h: 180, t: 70, dst: "ZACPT" },
  { name: "Sydney", lat: -33.86, lon: 151.2, h: 135, t: 60, dst: "AUSYD" },
  { name: "Mumbai", lat: 18.94, lon: 72.85, h: 270, t: 70, dst: "INBOM" },
  { name: "Hamburg", lat: 53.55, lon: 9.99, h: 45, t: 70, dst: "DEHAM" },
  { name: "Busan", lat: 35.1, lon: 129.04, h: 180, t: 70, dst: "KRPUS" },
  { name: "Vancouver", lat: 49.28, lon: -123.12, h: 315, t: 70, dst: "CAVAN" },
  { name: "Dubai Jebel Ali", lat: 25.01, lon: 55.06, h: 45, t: 80, dst: "AEJEA" },
  { name: "Gibraltar", lat: 36.14, lon: -5.35, h: 90, t: 70, dst: "GIB" },
  { name: "Malacca", lat: 2.5, lon: 101.8, h: 45, t: 80, dst: "MYMKZ" },
  { name: "North Atlantic", lat: 45.0, lon: -40.0, h: 270, t: 70, dst: "ATL" },
  { name: "South Pacific", lat: -20.0, lon: -140.0, h: 90, t: 70, dst: "PAC" },
  { name: "Indian Ocean", lat: -5.0, lon: 75.0, h: 180, t: 80, dst: "IND" },
];

export type ShipBbox = { minLat: number; maxLat: number; minLon: number; maxLon: number };

export const inShipBbox = (lat: number, lon: number, b: ShipBbox): boolean =>
  lat >= b.minLat && lat <= b.maxLat && lon >= b.minLon && lon <= b.maxLon;

export type RouteMarkerHit = {
  name: string;
  lat: number;
  lon: number;
  speed?: number;
  destination?: string;
  source: "route-marker";
};

export const portsInBbox = (ports: MedPortSeed[], bbox: ShipBbox | null): MedPortSeed[] => {
  if (!bbox) return ports;
  return ports.filter((p) => inShipBbox(p.lat, p.lon, bbox));
};

export const portsToRouteMarkers = (ports: MedPortSeed[]): RouteMarkerHit[] =>
  ports.map((p) => ({
    name: p.name,
    lat: p.lat,
    lon: p.lon,
    destination: p.dst,
    source: "route-marker" as const,
  }));
