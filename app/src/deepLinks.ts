/** True when the URL requests the SUPER SPORT sports-only tuner (?sport=1 or #supersport). */
export function readSuperSportDeepLink(): boolean {
  if (typeof window === "undefined") return false;
  const sport = new URLSearchParams(window.location.search).get("sport");
  if (sport === "1" || sport === "true" || sport === "yes") return true;
  const hash = window.location.hash.replace(/^#/, "").toLowerCase();
  return hash === "supersport" || hash === "sport" || hash === "super-sport";
}

/** True when URL requests the TV / live channels panel on load (?tv=1, #tv, or SUPER SPORT). */
export function readTvDeepLink(): boolean {
  if (typeof window === "undefined") return false;
  const tv = new URLSearchParams(window.location.search).get("tv");
  if (tv === "1" || tv === "true" || tv === "yes") return true;
  const hash = window.location.hash.replace(/^#/, "").toLowerCase();
  if (hash === "tv" || hash === "live" || hash === "live-tv") return true;
  return readSuperSportDeepLink();
}
