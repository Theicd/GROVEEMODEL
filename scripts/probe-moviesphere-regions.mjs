import { gunzipSync } from "node:zlib";
import https from "node:https";

function fetch(url) {
  return new Promise((res, rej) => {
    https
      .get(url, (r) => {
        if (r.statusCode >= 300 && r.statusCode < 400 && r.headers.location) {
          fetch(r.headers.location).then(res, rej);
          return;
        }
        const chunks = [];
        r.on("data", (c) => chunks.push(c));
        r.on("end", () => res(Buffer.concat(chunks)));
      })
      .on("error", rej);
  });
}

function findChannels(text, region) {
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*moviesphere[^<]*)<\/display-name>/gi;
  let m;
  while ((m = re.exec(text))) {
    console.log(`${region}: id=${m[1]} name=${m[2]}`);
  }
}

for (const region of ["us", "gb", "au", "de", "ca"]) {
  for (const ext of [".xml.gz", ".xml"]) {
    const url = `https://i.mjh.nz/SamsungTVPlus/${region}${ext}`;
    try {
      const buf = await fetch(url);
      let text;
      try {
        text = gunzipSync(buf).toString("utf8");
      } catch {
        text = buf.toString("utf8");
      }
      if (!text.includes("<?xml")) continue;
      findChannels(text, `${region}${ext}`);
      break;
    } catch {
      /* try next ext */
    }
  }
}
