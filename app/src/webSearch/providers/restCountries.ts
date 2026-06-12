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

type CapitalResponse = {
  error?: boolean;
  data?: { name: string; capital: string; iso2: string; iso3?: string };
};

type CurrencyResponse = {
  error?: boolean;
  data?: { name: string; currency: string; iso2: string; iso3?: string };
};

type PopulationResponse = {
  error?: boolean;
  data?: {
    country: string;
    populationCounts?: Array<{ year: string; value: number }>;
  };
};

const postCountriesNow = async <T>(path: string, country: string): Promise<T> =>
  fetchJson<T>(`https://countriesnow.space/api/v0.1/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ country: normalizeCountrySearchName(country).toLowerCase() }),
  });

export const resolveCountry = async (name: string): Promise<CountryRecord | null> => {
  const search = normalizeCountrySearchName(name);
  try {
    const [cap, cur, pop] = await Promise.all([
      postCountriesNow<CapitalResponse>("countries/capital", search),
      postCountriesNow<CurrencyResponse>("countries/currency", search).catch(() => null),
      postCountriesNow<PopulationResponse>("countries/population", search).catch(() => null),
    ]);

    if (cap.error || !cap.data?.iso2) return null;

    const latestPop = pop?.data?.populationCounts?.at(-1)?.value;

    return {
      name: cap.data.name,
      capital: cap.data.capital,
      iso2: cap.data.iso2,
      iso3: cap.data.iso3,
      currency: cur?.data?.currency,
      population: latestPop,
    };
  } catch {
    return null;
  }
};

const formatCountryBlock = (c: CountryRecord): string => {
  const lines = [
    `שם: ${c.name}`,
    `קוד: ${c.iso2}${c.iso3 ? ` / ${c.iso3}` : ""}`,
    `בירה: ${c.capital}`,
    ...(c.population ? [`אוכלוסיה (אומדן אחרון): ${c.population.toLocaleString("he-IL")}`] : []),
    ...(c.currency ? [`מטבע: ${c.currency}`] : []),
  ];
  return lines.join("\n");
};

export const fetchCountrySearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "rest-countries" as const;
  const label = "מדינות (CountriesNow)";
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
      url: `https://countriesnow.space`,
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
