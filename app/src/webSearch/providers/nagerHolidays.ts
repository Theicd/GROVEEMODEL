import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractCountryPhrase } from "../queryExtract";
import { resolveCountry } from "./restCountries";

type NagerHoliday = {
  date: string;
  localName: string;
  name: string;
  countryCode: string;
  fixed: boolean;
  global: boolean;
  types?: string[];
};

export const fetchHolidaySearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "nager-holidays" as const;
  const label = "חגים (Nager.Date)";
  try {
    const countryName = extractCountryPhrase(query);
    if (!countryName) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא זוהתה מדינה לחגים",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const country = await resolveCountry(countryName);
    if (!country?.iso2) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצאה מדינה: ${countryName}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const year = new Date().getFullYear();
    const holidays = await fetchJson<NagerHoliday[]>(
      `https://date.nager.at/api/v3/PublicHolidays/${year}/${country.iso2}`,
    );

    if (!holidays.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין נתוני חגים",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const today = new Date().toISOString().slice(0, 10);
    const todayHoliday = holidays.find((h) => h.date === today);
    const upcoming = holidays
      .filter((h) => h.date >= today)
      .slice(0, 8);

    const lines = [
      `מדינה: ${country.name ?? countryName} (${country.iso2}) · שנה ${year}`,
      todayHoliday
        ? `היום (${today}): כן — ${todayHoliday.localName} (${todayHoliday.name})`
        : `היום (${today}): לא חג ציבורי רשמי`,
      "חגים קרובים:",
      ...upcoming.map(
        (h) =>
          `- ${h.date}: ${h.localName} (${h.name})${h.global ? "" : " · מקומי"}`,
      ),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://date.nager.at/Country/${country.iso2}/${year}`,
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
