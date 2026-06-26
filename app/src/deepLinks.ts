/** True when URL requests the TV / live channels panel on load (?tv=1 or #tv). */
export function readTvDeepLink(): boolean {
  if (typeof window === "undefined") return false;
  const tv = new URLSearchParams(window.location.search).get("tv");
  if (tv === "1" || tv === "true" || tv === "yes") return true;
  const hash = window.location.hash.replace(/^#/, "").toLowerCase();
  return hash === "tv" || hash === "live" || hash === "live-tv";
}
