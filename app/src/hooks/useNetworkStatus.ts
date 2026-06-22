import { useCallback, useEffect, useState } from "react";
import {
  probeNetworkReachable,
  resolveNetworkReachability,
  type NetworkReachability,
} from "../networkReachability";

export type NetworkStatus = NetworkReachability;

const PROBE_INTERVAL_MS = 45_000;

export function useNetworkStatus(): NetworkStatus {
  const [status, setStatus] = useState<NetworkStatus>(() =>
    typeof navigator !== "undefined" && navigator.onLine ? "online" : "offline",
  );

  const refresh = useCallback(async () => {
    setStatus(await resolveNetworkReachability());
  }, []);

  useEffect(() => {
    void refresh();
    const onOnline = () => void refresh();
    const onOffline = () => setStatus("offline");
    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOffline);
    const id = window.setInterval(() => void refresh(), PROBE_INTERVAL_MS);
    return () => {
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOffline);
      window.clearInterval(id);
    };
  }, [refresh]);

  return status;
}

export { probeNetworkReachable as probeReachable };

export function networkStatusLabel(status: NetworkStatus, uiLang: "he" | "en" = "he"): string {
  if (uiLang === "he") {
    if (status === "online") return "מחובר לרשת";
    if (status === "limited") return "רשת מוגבלת";
    return "ללא חיבור";
  }
  if (status === "online") return "Online";
  if (status === "limited") return "Limited network";
  return "Offline";
}
