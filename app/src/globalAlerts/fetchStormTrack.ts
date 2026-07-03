import { fetchStormBriefing, type StormBriefing } from "./fetchStormBriefing";

export async function fetchStormTrack(
  eventId: number,
  episodeId: number,
): Promise<StormBriefing["track"] | null> {
  const b = await fetchStormBriefing(eventId, episodeId);
  return b?.track ?? null;
}

export { stormTrackCacheKey } from "./fetchStormBriefing";
