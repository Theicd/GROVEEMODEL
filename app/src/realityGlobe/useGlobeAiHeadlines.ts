import { useEffect, useRef, useState } from "react";
import type { GlobeIntelSnapshot } from "./intelFeed";
import { requestAiGlobeHeadlines } from "./globeHeadlineBridge";
import type { IntelHeadline } from "./intelFeed";
import type { UserGeoRegion } from "./geoLocation";

export function useGlobeAiHeadlines(
  active: boolean,
  snapshot: GlobeIntelSnapshot,
  geo: UserGeoRegion | null,
  modelReady: boolean,
) {
  const [aiHeadlines, setAiHeadlines] = useState<IntelHeadline[]>([]);
  const lastSigRef = useRef("");
  const busyRef = useRef(false);

  useEffect(() => {
    if (!active || !modelReady || !geo) return;

    const sig = snapshot.tickers
      .slice(0, 8)
      .map((t) => `${t.id}:${t.text}`)
      .join("|");
    if (!sig || sig === lastSigRef.current || busyRef.current) return;

    const timer = window.setTimeout(async () => {
      busyRef.current = true;
      try {
        const headlines = await requestAiGlobeHeadlines({
          tickers: snapshot.tickers,
          headlines: snapshot.headlines,
          countryCode: geo.countryCode,
          countryName: geo.countryName,
        });
        if (headlines.length) {
          lastSigRef.current = sig;
          setAiHeadlines(headlines);
        }
      } finally {
        busyRef.current = false;
      }
    }, 2500);

    return () => window.clearTimeout(timer);
  }, [active, modelReady, geo, snapshot.tickers, snapshot.headlines]);

  return aiHeadlines;
}
