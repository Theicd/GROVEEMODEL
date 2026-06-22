import type { ApiKeyProviderId } from "./apiKeyStore";

const USAGE_KEY = "grovee-api-usage-v1";
const ENABLED_KEY = "grovee-api-provider-enabled";

export const PROVIDER_ENABLED_EVENT = "grovee-api-provider-enabled-changed";
export const PROVIDER_USAGE_EVENT = "grovee-api-provider-usage-changed";

export type ProviderUsageRecord = {
  requestCount: number;
  successCount: number;
  totalHits: number;
  totalBytesApprox: number;
  lastRequestAt?: number;
  lastHitCount?: number;
  lastBytesApprox?: number;
  creditsRemaining?: number;
};

type UsageStore = Partial<Record<ApiKeyProviderId, ProviderUsageRecord>>;

const emptyRecord = (): ProviderUsageRecord => ({
  requestCount: 0,
  successCount: 0,
  totalHits: 0,
  totalBytesApprox: 0,
});

const readStore = (): UsageStore => {
  if (typeof localStorage === "undefined") return {};
  try {
    const raw = localStorage.getItem(USAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as UsageStore;
  } catch {
    return {};
  }
};

const writeStore = (store: UsageStore): void => {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(USAGE_KEY, JSON.stringify(store));
  window.dispatchEvent(new CustomEvent(PROVIDER_USAGE_EVENT));
};

export const isProviderEnabled = (id: ApiKeyProviderId): boolean => {
  if (typeof localStorage === "undefined") return true;
  const raw = localStorage.getItem(`${ENABLED_KEY}-${id}`);
  if (raw === null) return true;
  return raw === "1";
};

export const setProviderEnabled = (id: ApiKeyProviderId, enabled: boolean): void => {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(`${ENABLED_KEY}-${id}`, enabled ? "1" : "0");
  window.dispatchEvent(new CustomEvent(PROVIDER_ENABLED_EVENT));
};

export const getProviderUsage = (id: ApiKeyProviderId): ProviderUsageRecord => {
  const store = readStore();
  return { ...emptyRecord(), ...store[id] };
};

export const recordProviderUsage = (
  id: ApiKeyProviderId,
  input: {
    ok: boolean;
    hitCount?: number;
    bytesApprox?: number;
    creditsRemaining?: number;
  },
): ProviderUsageRecord => {
  const store = readStore();
  const prev = { ...emptyRecord(), ...store[id] };
  const hitCount = input.hitCount ?? 0;
  const bytes = input.bytesApprox ?? 0;

  const next: ProviderUsageRecord = {
    ...prev,
    requestCount: prev.requestCount + 1,
    successCount: prev.successCount + (input.ok ? 1 : 0),
    totalHits: prev.totalHits + (input.ok ? hitCount : 0),
    totalBytesApprox: prev.totalBytesApprox + (input.ok ? bytes : 0),
    lastRequestAt: Date.now(),
    lastHitCount: input.ok ? hitCount : prev.lastHitCount,
    lastBytesApprox: input.ok ? bytes : prev.lastBytesApprox,
    creditsRemaining:
      input.creditsRemaining != null ? input.creditsRemaining : prev.creditsRemaining,
  };

  store[id] = next;
  writeStore(store);
  return next;
};

export const resetProviderUsage = (id: ApiKeyProviderId): void => {
  const store = readStore();
  delete store[id];
  writeStore(store);
};

export const formatBytesKb = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  return kb < 1024 ? `${kb.toFixed(1)} KB` : `${(kb / 1024).toFixed(2)} MB`;
};

export const formatUsageSummaryHe = (
  id: ApiKeyProviderId,
  usage: ProviderUsageRecord,
): string => {
  const parts: string[] = [`${usage.requestCount} בקשות`, `${usage.successCount} הצליחו`];
  if (usage.lastHitCount != null) {
    parts.push(`אחרונה: ${usage.lastHitCount} תוצאות`);
  }
  if (usage.lastBytesApprox != null && usage.lastBytesApprox > 0) {
    parts.push(`~${formatBytesKb(usage.lastBytesApprox)}`);
  }
  if (id === "scavio" && usage.creditsRemaining != null) {
    parts.push(`נותרו ${usage.creditsRemaining} קרדיטים (Scavio)`);
  }
  return parts.join(" · ");
};
