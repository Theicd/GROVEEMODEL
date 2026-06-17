#!/usr/bin/env node
/** Stop GROVEEMODEL-main Vite (old «טען מודל מקומי» UI) so it cannot hijack dev ports. */
import { execSync } from "node:child_process";

function listNodeProcesses() {
  try {
    const out = execSync('wmic process where "name=\'node.exe\'" get ProcessId,CommandLine /format:list', {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    const blocks = out.split(/\r?\n\r?\n/);
    const rows = [];
    for (const block of blocks) {
      const cmdMatch = block.match(/CommandLine=(.+)/);
      const pidMatch = block.match(/ProcessId=(\d+)/);
      if (!cmdMatch || !pidMatch) continue;
      rows.push({ pid: pidMatch[1], cmd: cmdMatch[1].trim() });
    }
    return rows;
  } catch {
    return [];
  }
}

const victims = listNodeProcesses().filter(
  (p) => /GROVEEMODEL-main/i.test(p.cmd) && /vite/i.test(p.cmd),
);

for (const { pid, cmd } of victims) {
  try {
    execSync(`taskkill /PID ${pid} /F`, { stdio: "ignore" });
    console.warn("[kill-grovee-main-dev] stopped GROVEEMODEL-main vite PID " + pid);
    console.warn("  " + cmd.slice(0, 120) + (cmd.length > 120 ? "…" : ""));
  } catch {
    /* already gone */
  }
}
