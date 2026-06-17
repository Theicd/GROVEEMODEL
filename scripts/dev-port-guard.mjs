#!/usr/bin/env node
/**
 * Block/warn when GROVEEMODEL-main (old «טען מודל מקומי» UI) is running.
 */
import { execSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const primaryPort = process.env.GROVEE_DEV_PORT ?? "5180";
const scanPorts = [5173, 5174, 5180, 5181, 5182, 5183, 5184, 5185];

function getListeners() {
  try {
    const out = execSync("netstat -ano | findstr LISTENING", {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    const rows = [];
    for (const line of out.split(/\r?\n/)) {
      if (!line.includes("127.0.0.1:")) continue;
      const portMatch = line.match(/127\.0\.0\.1:(\d+)/);
      if (!portMatch) continue;
      const port = portMatch[1];
      if (!scanPorts.includes(port)) continue;
      const parts = line.trim().split(/\s+/);
      const pid = parts[parts.length - 1];
      if (pid) rows.push({ port, pid });
    }
    return rows;
  } catch {
    return [];
  }
}

function getProcessCommand(pid) {
  try {
    const out = execSync(`wmic process where "ProcessId=${pid}" get CommandLine /format:list`, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    const match = out.match(/CommandLine=(.+)/);
    return match?.[1]?.trim() ?? "";
  } catch {
    return "";
  }
}

const listeners = getListeners();
const mainServers = [];
const thisServers = [];

for (const { port, pid } of listeners) {
  const cmd = getProcessCommand(pid);
  if (!cmd) continue;
  if (/GROVEEMODEL-main/i.test(cmd)) mainServers.push({ port, pid, cmd });
  else if (/GROVEEMODEL/i.test(cmd) && /vite/i.test(cmd)) thisServers.push({ port, pid, cmd });
}

if (mainServers.some((s) => s.port === primaryPort)) {
  const s = mainServers.find((x) => x.port === primaryPort);
  console.error(
    [
      "",
      "  ✖ Port " + primaryPort + " is served by GROVEEMODEL-main (old UI: «טען מודל מקומי»).",
      "    PID " + s?.pid,
      "",
      "  Fix: close that terminal or run: taskkill /PID " + s?.pid + " /F",
      "  Then: cd " + root + " && npm run dev",
      "  Open: http://127.0.0.1:" + primaryPort + "/  (badge: HAL·SPACE)",
      "",
    ].join("\n"),
  );
  process.exit(1);
}

if (mainServers.length > 0) {
  console.warn(
    [
      "",
      "  ⚠ GROVEEMODEL-main (OLD intro UI) is also running:",
      ...mainServers.map((s) => "    → http://127.0.0.1:" + s.port + "/  (PID " + s.pid + ")"),
      "",
      "  Use ONLY: http://127.0.0.1:" + primaryPort + "/  from folder GROVEEMODEL",
      "  New UI signs: top badge «HAL·SPACE», button «טען מודל לדפדפן»",
      "",
    ].join("\n"),
  );
}

if (thisServers.some((s) => s.port === primaryPort)) {
  // Another GROVEEMODEL instance already on primary port — vite will fail or reuse; warn only.
  const existing = thisServers.find((s) => s.port === primaryPort);
  if (existing) {
    console.warn("[dev-port-guard] GROVEEMODEL already listening on " + primaryPort + " (PID " + existing.pid + ").");
  }
}
