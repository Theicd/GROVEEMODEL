async function q(label, query) {
  const u = `https://archive.org/advancedsearch.php?q=${encodeURIComponent(query)}&output=json&rows=5&fl[]=identifier&fl[]=title&fl[]=emulator&fl[]=downloads&fl[]=year&fl[]=reviews`;
  const d = await fetch(u).then((r) => r.json());
  console.log(`\n=== ${label} === ${d.response?.numFound} found`);
  console.log(JSON.stringify(d.response?.docs?.slice(0, 3), null, 2));
}

await q("PS2", '(subject:"PlayStation 2" OR subject:PS2) AND mediatype:software AND (emulator:*)');
await q("MK", "title:mortal AND title:kombat AND mediatype:software AND (emulator:*)");
await q("1987", "mediatype:software AND (emulator:*) AND year:1987");
await q("psx pool", "mediatype:software AND (emulator:psx OR emulator:ps1)");
