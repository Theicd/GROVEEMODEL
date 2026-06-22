import type { NetworkStatus } from "../hooks/useNetworkStatus";
import { networkStatusLabel } from "../hooks/useNetworkStatus";

type Props = {
  status: NetworkStatus;
  uiLang?: "he" | "en";
};

export function NetworkStatusIcon({ status, uiLang = "he" }: Props) {
  const label = networkStatusLabel(status, uiLang);
  const icon = status === "online" ? "📶" : status === "limited" ? "⚠️" : "📵";

  return (
    <span
      className={`network-status network-status--${status}`}
      title={label}
      aria-label={label}
      role="status"
    >
      <span className="network-status-icon" aria-hidden="true">
        {icon}
      </span>
      <span className="network-status-label">{label}</span>
    </span>
  );
}
