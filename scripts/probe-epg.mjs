async function gunzip(buf) {
  if (typeof DecompressionStream !== "undefined") {
    const ds = new DecompressionStream("gzip");
    return await new Response(new Blob([buf]).stream().pipeThrough(ds)).text();
  }
  const { gunzipSync } = await import("node:zlib");
  return gunzipSync(Buffer.from(buf)).toString("utf8");
}

const res = await fetch("https://i.mjh.nz/all/epg.xml.gz");
const xml = await gunzip(await res.arrayBuffer());
console.log("xml bytes", xml.length);
const copsCh = xml.match(/<channel id="([^"]+)"[^>]*>\s*<display-name>COPS<\/display-name>/);
console.log("COPS channel", copsCh?.[1]);
if (copsCh?.[1]) {
  const id = copsCh[1];
  const count = (xml.match(new RegExp(`programme channel="${id}"`, "g")) || []).length;
  console.log("COPS programmes", count);
  const sample = xml.match(new RegExp(`<programme channel="${id}" start="[^"]+" stop="[^"]+">\\s*<title>([^<]+)</title>`));
  console.log("sample title", sample?.[1]);
}
