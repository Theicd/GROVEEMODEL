/**
 * Live QA — PeerTube (Sepia), Internet Archive, optional Invidious.
 * Run: npm run test:federated
 */
import { describe, expect, it } from "vitest";

import { mergeSourcesToHits } from "../searchResults/mergeSearchHits";
import { fetchInternetArchiveMediaSearch } from "./providers/internetArchiveMedia";
import { fetchPeerTubeVideosSearch } from "./providers/peertubeMedia";
import { fetchInvidiousVideosSearch } from "./providers/invidiousMedia";

const hasPlayableVideo = (hits: ReturnType<typeof mergeSourcesToHits>) =>
  hits.some((h) => h.kind === "video" && (h.mediaPlayUrl?.includes(".mp4") || h.mediaEmbedMode));

describe("federated video live QA", () => {
  it("PeerTube (Sepia) returns playable video hits", async () => {
    const result = await fetchPeerTubeVideosSearch("linux tutorial");
    expect(result.ok, result.error).toBe(true);
    expect(result.mediaHits?.length).toBeGreaterThan(0);

    const merged = mergeSourcesToHits([result]);
    expect(merged[0].sourceLabel).toBe("PeerTube");
    expect(merged[0].imageUrl).toBeTruthy();
    expect(merged[0].mediaPlayUrl).toBeTruthy();
  }, 30_000);

  it("Internet Archive returns Hebrew-friendly playable archive videos", async () => {
    const result = await fetchInternetArchiveMediaSearch("Israel television");
    if (!result.ok) {
      // Archive may be slow — try a broader English query
      const fallback = await fetchInternetArchiveMediaSearch("documentary");
      expect(fallback.ok, fallback.error).toBe(true);
      const hits = mergeSourcesToHits([fallback]);
      expect(hasPlayableVideo(hits)).toBe(true);
      return;
    }
    const hits = mergeSourcesToHits([result]);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].provider).toBe("internet-archive-media");
    expect(hits[0].mediaPlayUrl).toMatch(/archive\.org\/download\//);
  }, 45_000);

  it("Invidious returns hits or fails gracefully without throwing", async () => {
    const result = await fetchInvidiousVideosSearch("linux");
    if (result.ok) {
      const hits = mergeSourcesToHits([result]);
      expect(hits[0].mediaEmbedMode).toBe(true);
      expect(hits[0].mediaPlayUrl).toContain("/embed/");
    } else {
      expect(result.error).toBeTruthy();
    }
  }, 30_000);

  it("orchestrator panel search wires federated video providers", async () => {
    const { runWebSearch } = await import("./orchestrator");
    const result = await runWebSearch("linux tutorial video", {
      panelSearch: true,
      plan: { useWebFallback: false, blendNewsWithWeb: false, queries: ["linux tutorial video"] },
    });
    const providers = result.sources.map((s) => s.provider);
    expect(providers).toContain("peertube-videos");
    expect(providers).toContain("internet-archive-media");
    expect(providers).toContain("invidious-videos");
    const okVideo = result.sources.filter(
      (s) =>
        s.ok &&
        (s.provider === "peertube-videos" ||
          s.provider === "internet-archive-media" ||
          s.provider === "pixabay-videos"),
    );
    expect(okVideo.length).toBeGreaterThan(0);
  }, 60_000);
});
