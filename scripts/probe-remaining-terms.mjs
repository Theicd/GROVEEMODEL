import { gunzipSync } from "node:zlib";
const terms = ["charge", "discoverfilm", "filmdetective", "film detective", "positiv", "rally", "deluxe", "whiplash", "cipher", "tunebox", "tom jerry", "classique", "groovy", "mix hollywood", "30a tv", "autentic", "inwild", "history hunters", "chat show", "teen nick"];
const sources = [["plex","https://i.mjh.nz/Plex/us.xml.gz"],["roku","https://i.mjh.nz/Roku/all.xml.gz"],["samsung","https://i.mjh.nz/SamsungTVPlus/us.xml.gz"],["pluto","https://i.mjh.nz/PlutoTV/us.xml.gz"]];
for (const [k,u] of sources) {
  const xml = gunzipSync(Buffer.from(await (await fetch(u)).arrayBuffer())).toString("utf8");
  console.log("\n==",k);
  const re=/<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g; let m;
  while((m=re.exec(xml))) for(const t of terms) if(m[2].toLowerCase().includes(t)) console.log(t+":",m[2]);
