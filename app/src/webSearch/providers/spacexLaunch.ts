import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

type LaunchResult = {
  results?: Array<{
    name?: string;
    net?: string;
    status?: { name?: string; abbrev?: string };
    launch_service_provider?: { name?: string };
    rocket?: { configuration?: { full_name?: string } };
  }>;
};

/** Upcoming SpaceX launches — Launch Library 2 (free, no key). */
export const fetchSpaceXLaunchSearch = async (_query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "spacex-launches" as const;
  const label = "שיגורי SpaceX (Launch Library)";

  try {
    const data = await fetchJson<LaunchResult>(
      "https://ll.thespacedevs.com/2.0.0/launch/upcoming/?search=SpaceX&limit=5",
      undefined,
      { timeoutMs: 14_000 },
    );
    const launches = data.results ?? [];
    if (!launches.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין שיגורים קרובים ברשימה",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      "שיגורי SpaceX קרובים (Launch Library 2):",
      ...launches.map((l, i) => {
        const when = l.net ? new Date(l.net).toISOString().replace("T", " ").slice(0, 16) + " UTC" : "—";
        const rocket = l.rocket?.configuration?.full_name ?? "—";
        const status = l.status?.name ?? l.status?.abbrev ?? "—";
        return `${i + 1}. ${l.name ?? "—"} · ${when} · ${rocket} · ${status}`;
      }),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: "https://ll.thespacedevs.com/2.0.0/launch/upcoming/?search=SpaceX",
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
