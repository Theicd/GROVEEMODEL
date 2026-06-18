import { describe, expect, it } from "vitest";

import { mergeSourcesToHits } from "../searchResults/mergeSearchHits";
import type { SearchSourceResult } from "./types";

describe("federated video merge", () => {
  it("maps PeerTube, IA, and Invidious media to video hits with correct providers", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "peertube-videos",
        label: "PeerTube",
        ok: true,
        text: "",
        latencyMs: 50,
        mediaHits: [
          {
            id: "pt-1",
            mediaType: "video",
            title: "Open video",
            url: "https://peertube.example/v/1",
            playUrl: "https://peertube.example/dl/1.mp4",
            thumbnail: "https://peertube.example/t.jpg",
            source: "PeerTube",
            durationSec: 60,
          },
        ],
      },
      {
        provider: "internet-archive-media",
        label: "IA",
        ok: true,
        text: "",
        latencyMs: 60,
        mediaHits: [
          {
            id: "ia-1",
            mediaType: "video",
            title: "ארכיון ישראלי",
            url: "https://archive.org/details/x",
            playUrl: "https://archive.org/download/x/a.mp4",
            thumbnail: "https://archive.org/services/img/x",
            source: "Internet Archive",
          },
        ],
      },
      {
        provider: "invidious-videos",
        label: "Invidious",
        ok: true,
        text: "",
        latencyMs: 70,
        mediaHits: [
          {
            id: "inv-1",
            mediaType: "video",
            title: "YouTube mirror",
            url: "https://inv.example/watch?v=abc",
            playUrl: "https://inv.example/embed/abc",
            thumbnail: "https://img/abc.jpg",
            source: "YouTube",
            youtubeSubType: "video",
          },
        ],
      },
    ];

    const hits = mergeSourcesToHits(sources);
    const peertube = hits.find((h) => h.provider === "peertube-videos");
    const archive = hits.find((h) => h.provider === "internet-archive-media");
    const invidious = hits.find((h) => h.provider === "invidious-videos");

    expect(peertube?.kind).toBe("video");
    expect(peertube?.score).toBeGreaterThanOrEqual(62);
    expect(archive?.kind).toBe("video");
    expect(archive?.score).toBeGreaterThanOrEqual(66);
    expect(invidious?.kind).toBe("youtube");
    expect(invidious?.mediaEmbedMode).toBe(true);
  });
});
