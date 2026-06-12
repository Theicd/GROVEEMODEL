import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractCountryPhrase } from "../queryExtract";
import { resolveCountry } from "./restCountries";

type WbSearchHit = {
  id: string;
  label: string;
  description?: string;
};

type WbSearchResponse = {
  search?: WbSearchHit[];
};

type SparqlBinding = {
  personLabel?: { value: string };
  roleLabel?: { value: string };
};

type SparqlResponse = {
  results?: { bindings?: SparqlBinding[] };
};

const WIKIDATA_HEADERS = {
  Accept: "application/sparql-results+json",
  "User-Agent": "GROVEEMODEL/1.0 (browser chat; web search)",
};

/** Avoid wbsearchentities rate limits when ISO2 is known from CountriesNow. */
const ISO2_TO_WIKIDATA_QID: Record<string, string> = {
  IL: "Q801",
  DE: "Q183",
  FR: "Q142",
  US: "Q30",
  GB: "Q145",
  JP: "Q17",
  CN: "Q148",
  BR: "Q155",
  CA: "Q16",
  MX: "Q96",
  RU: "Q159",
  AU: "Q408",
  ES: "Q29",
  IT: "Q38",
  IN: "Q668",
  EG: "Q79",
  TR: "Q43",
  PL: "Q36",
  SE: "Q34",
  NO: "Q20",
  FI: "Q33",
  BE: "Q31",
  NL: "Q55",
  NZ: "Q664",
};

const findCountryQid = async (name: string, iso2?: string): Promise<string | null> => {
  if (iso2) {
    const mapped = ISO2_TO_WIKIDATA_QID[iso2.toUpperCase()];
    if (mapped) return mapped;
  }

  const data = await fetchJson<WbSearchResponse>(
    `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=${encodeURIComponent(name)}&language=en&format=json&origin=*&limit=8&type=item`,
  );
  const hit =
    data.search?.find((s) => /country|מדינה|nation|sovereign state/i.test(s.description ?? "")) ??
    data.search?.find((s) => /^Q\d+$/.test(s.id) && s.label.toLowerCase() === name.toLowerCase()) ??
    data.search?.[0];
  return hit?.id ?? null;
};

const fetchLeaders = async (countryQid: string): Promise<SparqlBinding[]> => {
  const sparql = `
SELECT ?personLabel ?roleLabel WHERE {
  {
    wd:${countryQid} wdt:P6 ?person .
    BIND("ראש ממשלה" AS ?roleLabel)
  }
  UNION
  {
    wd:${countryQid} wdt:P35 ?person .
    BIND("ראש מדינה / נשיא" AS ?roleLabel)
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "he,en". }
}`.trim();

  const url = `https://query.wikidata.org/sparql?format=json&query=${encodeURIComponent(sparql)}`;
  const data = await fetchJson<SparqlResponse>(url, { headers: WIKIDATA_HEADERS });
  return data.results?.bindings ?? [];
};

export const fetchGovernmentSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "wikidata-gov" as const;
  const label = "ממשל (Wikidata)";
  try {
    const countryName = extractCountryPhrase(query);
    if (!countryName) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא זוהתה מדינה",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const country = await resolveCountry(countryName);
    const lookupName = country?.name ?? countryName;
    const qid = await findCountryQid(lookupName, country?.iso2);
    if (!qid) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצאה ישות Wikidata: ${lookupName}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const leaders = await fetchLeaders(qid);
    if (!leaders.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצאו נתוני ממשל עדכניים",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      `מדינה: ${lookupName} (Wikidata ${qid})`,
      "נושאי משרה (Wikidata):",
      ...leaders.map((b) => `- ${b.personLabel?.value ?? "—"} · ${b.roleLabel?.value ?? "תפקיד"}`),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://www.wikidata.org/wiki/${qid}`,
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
