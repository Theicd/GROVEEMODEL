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

type WbSearchHit = { id: string; label: string; description?: string };
type WbSearchResponse = { search?: WbSearchHit[] };

type SparqlBinding = {
  countryLabel?: { value: string };
  capitalLabel?: { value: string };
  population?: { value: string };
  currencyLabel?: { value: string };
  iso2?: { value: string };
};

type SparqlResponse = { results?: { bindings?: SparqlBinding[] } };

const WIKIDATA_HEADERS = {
  Accept: "application/sparql-results+json",
  "User-Agent": "GROVEEMODEL/1.0 (browser chat; web search)",
};

const ISO2_HINTS: Record<string, string> = {
  germany: "DE",
  גרמניה: "DE",
  israel: "IL",
  ישראל: "IL",
  japan: "JP",
  יפן: "JP",
  brazil: "BR",
  ברזיל: "BR",
  canada: "CA",
  קנדה: "CA",
  france: "FR",
  צרפת: "FR",
  "united states": "US",
  "ארצות הברית": "US",
  usa: "US",
  britain: "GB",
  "united kingdom": "GB",
  בריטניה: "GB",
};

const isRestCountryArray = (data: unknown): data is RestCountry[] =>
  Array.isArray(data) && data.length > 0 && typeof data[0] === "object";

const fetchRestCountry = async (name: string): Promise<RestCountry | null> => {
  const search = encodeURIComponent(normalizeCountrySearchName(name));
  const apiKey = (import.meta.env.VITE_RESTCOUNTRIES_API_KEY as string | undefined)?.trim();
  if (apiKey) {
    try {
      const data = await fetchJson<{ data?: RestCountry[] }>(
        `https://restcountries.com/v5/name/${search}?fields=name,capital,cca2,cca3,currencies,population`,
        { headers: { Authorization: `Bearer ${apiKey}` } },
      );
      return data.data?.[0] ?? null;
    } catch {
      /* fall through to Wikidata */
    }
  }

  try {
    const list = await fetchJson<unknown>(
      `https://restcountries.com/v3.1/name/${search}?fields=name,capital,cca2,cca3,currencies,population`,
    );
    if (isRestCountryArray(list)) return list[0];
  } catch {
    /* deprecated v3 — use Wikidata */
  }
  return null;
};

const findCountryQid = async (name: string, iso2Hint?: string): Promise<string | null> => {
  const ISO2_TO_QID: Record<string, string> = {
    IL: "Q801", DE: "Q183", FR: "Q142", US: "Q30", GB: "Q145", JP: "Q17",
    CN: "Q148", BR: "Q155", CA: "Q16", MX: "Q96", RU: "Q159", AU: "Q408",
    ES: "Q29", IT: "Q38", IN: "Q668", EG: "Q79", TR: "Q43", PL: "Q36",
  };
  if (iso2Hint && ISO2_TO_QID[iso2Hint.toUpperCase()]) {
    return ISO2_TO_QID[iso2Hint.toUpperCase()];
  }
  const search = encodeURIComponent(normalizeCountrySearchName(name));
  const data = await fetchJson<WbSearchResponse>(
    `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=${search}&language=en&format=json&origin=*&type=item&limit=5`,
  );
  const hit = data.search?.find((h) => /country|sovereign|state|nation|מדינה/i.test(h.description ?? ""))
    ?? data.search?.[0];
  return hit?.id ?? null;
};

const fetchCountryFromWikidata = async (name: string): Promise<CountryRecord | null> => {
  const key = normalizeCountrySearchName(name).toLowerCase();
  const isoHint = ISO2_HINTS[key] ?? ISO2_HINTS[name.trim().toLowerCase()];
  const qid = await findCountryQid(name, isoHint);
  if (!qid) return null;

  const sparql = `
SELECT ?countryLabel ?capitalLabel ?population ?currencyLabel ?iso2 WHERE {
  BIND(wd:${qid} AS ?country)
  OPTIONAL { ?country wdt:P36 ?capital . ?capital rdfs:label ?capitalLabel . FILTER(LANG(?capitalLabel) IN ("he","en")) }
  OPTIONAL { ?country wdt:P1082 ?population . }
  OPTIONAL { ?country wdt:P38 ?currency . ?currency rdfs:label ?currencyLabel . FILTER(LANG(?currencyLabel) IN ("he","en")) }
  OPTIONAL { ?country wdt:P297 ?iso2 . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "he,en". }
} LIMIT 1`;

  const url = `https://query.wikidata.org/sparql?query=${encodeURIComponent(sparql)}&format=json`;
  const data = await fetchJson<SparqlResponse>(url, { headers: WIKIDATA_HEADERS }, { timeoutMs: 14_000 });
  const row = data.results?.bindings?.[0];
  if (!row) return null;

  return {
    name: row.countryLabel?.value ?? name,
    capital: row.capitalLabel?.value ?? "—",
    iso2: row.iso2?.value ?? isoHint ?? "—",
    currency: row.currencyLabel?.value,
    population: row.population?.value ? Number(row.population.value) : undefined,
  };
};

export const resolveCountry = async (name: string): Promise<CountryRecord | null> => {
  const row = await fetchRestCountry(name);
  if (row?.cca2) {
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
  }
  return fetchCountryFromWikidata(name);
};

const formatCountryBlock = (c: CountryRecord): string => {
  const lines = [
    `שם: ${c.name}`,
    `קוד: ${c.iso2}${c.iso3 ? ` / ${c.iso3}` : ""}`,
    `בירה: ${c.capital}`,
    ...(c.population ? [`אוכלוסיה: ${c.population.toLocaleString("he-IL")}`] : []),
    ...(c.currency ? [`מטבע: ${c.currency}`] : []),
    "מקור: Wikidata / REST Countries",
  ];
  return lines.join("\n");
};

export const fetchCountrySearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "rest-countries" as const;
  const label = "מדינות (Wikidata / REST Countries)";
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
      url: `https://www.wikidata.org/wiki/${country.iso2}`,
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
