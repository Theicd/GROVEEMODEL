const headers = {
  Accept: "application/sparql-results+json",
  "User-Agent": "GROVEEMODEL/1.0 (web search test)",
};

async function run(label, sparql) {
  const url = `https://query.wikidata.org/sparql?format=json&query=${encodeURIComponent(sparql.trim())}`;
  const r = await fetch(url, { headers });
  const d = await r.json();
  console.log(label, r.status, JSON.stringify(d.results?.bindings ?? d));
}

await run(
  "wdt P6",
  `SELECT ?person ?personLabel WHERE { wd:Q801 wdt:P6 ?person . SERVICE wikibase:label { bd:serviceParam wikibase:language "en,he". } }`,
);

await run(
  "position held",
  `SELECT ?person ?personLabel WHERE {
    wd:Q801 p:P35 ?stmt .
    ?stmt ps:P642 ?person .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en,he". }
  } LIMIT 5`,
);

await run(
  "office head of government",
  `SELECT ?person ?personLabel WHERE {
    wd:Q801 wdt:P6 ?office .
    ?office wdt:P1308 ?person .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en,he". }
  }`,
);
