import { fetchJson } from "../webSearch/fetchJson";

const NOMINATIM_HEADERS = {
  Accept: "application/json",
  "User-Agent": "GROVEEMODEL/1.0 (global alerts; contact: none)",
};

type ReverseResult = {
  display_name?: string;
  address?: {
    country?: string;
    state?: string;
    region?: string;
    ocean?: string;
    sea?: string;
  };
};

const regionCache = new Map<string, string>();

export async function reverseRegionLabel(lat: number, lon: number): Promise<string> {
  const key = `${lat.toFixed(1)},${lon.toFixed(1)}`;
  const cached = regionCache.get(key);
  if (cached) return cached;

  try {
    const data = await fetchJson<ReverseResult>(
      `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&accept-language=he&zoom=5`,
      { headers: NOMINATIM_HEADERS },
      { timeoutMs: 9_000 },
    );
    const a = data.address;
    let label = "";
    if (a?.country) {
      label = [a.state, a.region, a.country].filter(Boolean).join(" · ");
    } else if (a?.ocean || a?.sea) {
      label = `מעל ${a.ocean ?? a.sea}`;
    } else if (data.display_name) {
      label = data.display_name.split(",").slice(0, 3).join(" · ");
    } else {
      label = "אזור ים פתוח";
    }
    regionCache.set(key, label);
    return label;
  } catch {
    return "אזור ים פתוח";
  }
}

export function formatPopulationHe(n: number): string {
  if (n >= 1_000_000_000) return `~${(n / 1_000_000_000).toFixed(1)} מיליארד`;
  if (n >= 1_000_000) return `~${(n / 1_000_000).toFixed(1)} מיליון`;
  if (n >= 1_000) return `~${Math.round(n / 1_000)}K`;
  return String(n);
}
