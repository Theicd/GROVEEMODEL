import { gunzipSync } from "node:zlib";

const xml = gunzipSync(
  Buffer.from(await (await fetch("https://i.mjh.nz/SamsungTVPlus/us.xml.gz")).arrayBuffer()),
).toString("utf8");

const idx = xml.indexOf("Eye See You");
console.log(xml.slice(idx - 200, idx + 400));

const allLengths = xml.match(/<length[^>]*>[^<]+<\/length>/gi);
console.log("\nTotal length tags in file:", allLengths?.length ?? 0);
console.log("Samples:", allLengths?.slice(0, 8));
