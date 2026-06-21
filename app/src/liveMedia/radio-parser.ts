import type { RadioStation } from "./types";

interface RawStation {
  stationuuid: string;
  name: string;
  url: string;
  url_resolved: string;
  favicon: string;
  tags: string;
  country: string;
  countrycode: string;
  language: string;
  votes: number;
  codec: string;
  bitrate: number;
}

function hashId(str: string): string {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h << 5) - h + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h).toString(36);
}

export function parseRadioStations(data: RawStation[]): RadioStation[] {
  const out: RadioStation[] = [];
  for (const r of data) {
    if (!r.url_resolved && !r.url) continue;
    const stream = r.url_resolved || r.url;
    out.push({
      id: hashId(stream + r.name),
      name: r.name?.trim() || "Unknown Station",
      favicon: r.favicon || "",
      tags: r.tags ? r.tags.split(",").map((t) => t.trim()).filter(Boolean) : [],
      country: r.country || "",
      countrycode: (r.countrycode || "").toLowerCase(),
      language: (r.language || "").toLowerCase().split(",")[0] || "",
      stream,
      type: "radio",
      status: "unknown",
      lastCheck: 0,
      bitrate: r.bitrate,
      codec: r.codec,
      votes: r.votes,
      favorite: false,
      addedAt: Date.now(),
    });
  }
  return out;
}
