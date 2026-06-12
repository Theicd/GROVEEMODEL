async function post(path, body) {
  const r = await fetch(`https://countriesnow.space/api/v0.1/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  console.log(path, await r.json());
}

await post("countries/capital", { country: "israel" });
await post("countries/currency", { country: "germany" });
await post("countries/population", { country: "germany" });
