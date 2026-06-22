/** Tunables for Grove Search Companion (OpenSERP) — keep stable defaults here. */

/** All engines available in local OpenSERP install (see /health). */
export const COMPANION_ALL_ENGINES = [
  "google",
  "bing",
  "duckduckgo",
  "yandex",
  "ecosia",
] as const;

export const COMPANION_WEB_ENGINES = COMPANION_ALL_ENGINES.join(",");

/** Image megasearch — prefer engines that tolerate automation (Google often CAPTCHA). */
export const COMPANION_IMAGE_ENGINES = "bing,duckduckgo,google,ecosia";

export const COMPANION_DEFAULT_LANG = "HE";
export const COMPANION_DEFAULT_REGION = "IL";

export const COMPANION_WEB_LIMIT = 12;
export const COMPANION_IMAGE_LIMIT = 8;

/** Parallel merge across engines — best coverage; mega_timeout in config.yaml is 90s. */
export const COMPANION_WEB_MODE = "balanced" as const;

/** Fast first-success for images (Bing image search is usually reliable). */
export const COMPANION_IMAGE_MODE = "fast" as const;

export const COMPANION_WEB_TIMEOUT_MS = 55_000;
export const COMPANION_IMAGE_TIMEOUT_MS = 35_000;
export const COMPANION_PROBE_TIMEOUT_MS = 25_000;
