#!/usr/bin/env node
/**
 * Free GROVEEMODEL dev port before starting Vite (prevents EADDRINUSE / strictPort crash).
 * English-only output for BAT compatibility.
 */
import { execSync } from "node:child_process";

const primaryPort = process.env.GROVEE_DEV_PORT ?? "5180";

function getListenersOnPort(port) {
  try {
    const out = execSync("netstat -ano", {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    const pids = new Set();
    for (const line of out.split(/\r?\n/)) {
      if (!line.includes("LISTENING")) continue;
      if (!line.includes(`127.0.0.1:${port}`) && !line.includes(`0.0.0.0:${port}`)) continue;
      const parts = line.trim().split(/\s+/);
      const pid = parts[parts.length - 1];
      if (pid && /^\d+$/.test(pid) && pid !== "0") pids.add(pid);
    }
    return [...pids];
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

function killPid(pid, reason) {
  try {
    execSync(`taskkill /PID ${pid} /F /T`, { stdio: "ignore" });
    console.log(`[kill-grovee-dev-port] stopped PID ${pid} (${reason})`);
    return true;
  } catch {
    return false;
  }
}

const pids = getListenersOnPort(primaryPort);
if (!pids.length) {
  console.log(`[kill-grovee-dev-port] port ${primaryPort} is free`);
  process.exit(0);
}

let stopped = 0;
for (const pid of pids) {
  const cmd = getProcessCommand(pid);
  const label = cmd ? cmd.slice(0, 100) : "unknown process";
  if (killPid(pid, label)) stopped += 1;
}

if (stopped > 0) {
  try {
    execSync("timeout /t 1 /nobreak >nul", { stdio: "ignore", shell: true });
  } catch {
    /* non-Windows or timeout missing */
  }
}

const remaining = getListenersOnPort(primaryPort);
if (remaining.length) {
  console.warn(
    `[kill-grovee-dev-port] port ${primaryPort} still in use (PIDs: ${remaining.join(", ")}). Close manually if dev fails.`,
  );
  process.exit(1);
}

console.log(`[kill-grovee-dev-port] port ${primaryPort} ready`);
