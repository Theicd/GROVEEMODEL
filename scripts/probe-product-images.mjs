const bc = process.argv[2] || "7290000651620";
const chains = ["carrefour", "rami-levy", "victory", "shufersal", "mega", "yohananof"];
for (const c of chains) {
  const r = await fetch(`https://price-api.additlist.com/images/catalog/${c}/${bc}.jpg`, { method: "HEAD" });
  console.log("additlist", c, r.status);
}
for (const [name, url] of [
  ["shufersal", `https://img.shufersal.co.il/imgs/Products_Vertical/${bc}_V_large.jpg`],
  ["rami", `https://static.rfrsh.co.il/supermarket/product/${bc}/small.jpg`],
]) {
  try {
    const r = await fetch(url, { method: "HEAD" });
    console.log(name, r.status);
  } catch (e) {
    console.log(name, "err", e.cause?.code || e.message);
  }
}
try {
  const html = await (await fetch(`https://cheapersal.co.il/product/${bc}`)).text();
  const og = html.match(/property="og:image"\s+content="([^"]+)"/i)?.[1];
  console.log("og", og?.slice(0, 120));
  const addit = html.match(/https:\/\/price-api\.additlist\.com\/images\/[^"'\s<>]+/i)?.[0];
  console.log("addit-in-html", addit);
  const price = html.match(/המחיר הזול[^₪]{0,48}₪([\d.]+)/)?.[1];
  console.log("price", price);
} catch (e) {
  console.log("cheapersal err", e.message);
}
