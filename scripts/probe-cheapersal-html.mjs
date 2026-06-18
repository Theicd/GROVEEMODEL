const bc = process.argv[2] || "7290004131074";
const url = `https://cheapersal.co.il/product/${bc}`;
const html = await fetch(url, {
  headers: { "User-Agent": "Mozilla/5.0 (compatible; GROVEEMODEL/1.0)" },
}).then((r) => r.text());
console.log("url", url);
console.log("title", html.match(/<title>([^<]+)/)?.[1]);
console.log("og:image", html.match(/property="og:image"\s+content="([^"]+)"/i)?.[1]);
console.log("additlist", html.match(/https:\/\/price-api\.additlist\.com\/images\/[^"'\s<>]+/i)?.[0]);
const prices = [...html.matchAll(/₪\s*([\d]+(?:\.[\d]{1,2})?)/g)].slice(0, 8).map((m) => m[0]);
console.log("prices", prices);
console.log("json-ld", html.includes("application/ld+json"));
const ldBlocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/gi)];
for (const [, block] of ldBlocks) {
  try {
    const j = JSON.parse(block);
    console.log("ld parsed", JSON.stringify(j, null, 2).slice(0, 1200));
  } catch (e) {
    console.log("ld parse err", String(e));
  }
}
const next = html.match(/<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/i)?.[1];
if (next) {
  const j = JSON.parse(next);
  console.log("next keys", Object.keys(j));
  console.log("next sample", JSON.stringify(j.props?.pageProps ?? j).slice(0, 800));
}
console.log("המחיר הזול", html.match(/המחיר הזול[\s\S]{0,80}/)?.[0]);
