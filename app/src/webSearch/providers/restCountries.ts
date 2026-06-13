import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractCountryPhrase, normalizeCountrySearchName } from "../queryExtract";

export type CountryRecord = {
  name: string;
  capital: string;
  iso2: string;
  iso3?: string;
  currency?: string;
  population?: number;
};

type RestCountry = {
  name?: { common?: string; official?: string };
  capital?: string[];
  cca2?: string;
  cca3?: string;
  currencies?: Record<string, { name?: string; symbol?: string }>;
  population?: number;
};

const fetchRestCountry = async (name: string): Promise<RestCountry | null> => {
  const search = encodeURIComponent(normalizeCountrySearchName(name));
  try {
    const list = await fetchJson<RestCountry[]>(
      `https://restcountries.com/v3.1/name/${search}?fields=name,capital,cca2,cca3,currencies,population`,
    );
    return list[0] ?? null;
  } catch {
    try {
      const list = await fetchJson<RestCountry[]>(
        `https://restcountries.com/v3.1/translation/${search}?fields=name,capital,cca2,cca3,currencies,population`,
      );
      return list[0] ?? null;
    } catch {
      return null;
    }
  }
};

export const resolveCountry = async (name: string): Promise<CountryRecord | null> => {
  const row = await fetchRestCountry(name);
  if (!row?.cca2) return null;

  const currencyCode = row.currencies ? Object.keys(row.currencies)[0] : undefined;
  const currencyName = currencyCode && row.currencies?.[currencyCode]?.name;

  return {
    name: row.name?.common ?? name,
    capital: row.capital?.[0] ?? "—",
    iso2: row.cca2,
    iso3: row.cca3,
    currency: currencyName ? `${currencyName} (${currencyCode})` : currencyCode,
    population: row.population,
  };
};

const formatCountryBlock = (c: CountryRecord): string => {
  const lines = [
    `שם: ${c.name}`,
    `קוד: ${c.iso2}${c.iso3 ? ` / ${c.iso3}` : ""}`,
    `בירה: ${c.capital}`,
    ...(c.population ? [`אוכלוסיה: ${c.population.toLocaleString("he-IL")}`] : []),
    ...(c.currency ? [`מטבע: ${c.currency}`] : []),
  ];
  return lines.join("\n");
};

export const fetchCountrySearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "rest-countries" as const;
  const label = "מדינות (REST Countries)";
  try {
    const countryName = extractCountryPhrase(query) ?? query.trim();
    if (countryName.length < 2) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא זוהתה מדינה בשאלה",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const country = await resolveCountry(countryName);
    if (!country) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצאה מדינה: ${countryName}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    return {
      provider,
      label,
      ok: true,
      text: formatCountryBlock(country),
      url: `https://restcountries.com/v3.1/name/${encodeURIComponent(country.name)}`,
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

export { formatCountryBlock };
