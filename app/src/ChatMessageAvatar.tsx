import { GroveeLogoMark } from "./GroveeLogoMark";

type ChatMessageAvatarProps = {
  role: "user" | "assistant";
  /** HAL camera mode — subtle cyan accent instead of default assistant styling. */
  variant?: "default" | "hal";
  className?: string;
};

function UserAvatarIcon() {
  return (
    <svg className="msg-avatar-icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <circle cx="10" cy="7" r="3.25" fill="currentColor" opacity="0.92" />
      <path
        d="M4 17.5c0-3.038 2.686-5.5 6-5.5s6 2.462 6 5.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        opacity="0.88"
      />
    </svg>
  );
}

/** Chat row avatar — GroVee rings for assistant, silhouette for user. */
export function ChatMessageAvatar({ role, variant = "default", className = "" }: ChatMessageAvatarProps) {
  if (role === "user") {
    return (
      <div
        className={`msg-avatar msg-avatar--user ${className}`.trim()}
        aria-hidden="true"
      >
        <UserAvatarIcon />
      </div>
    );
  }

  const assistantClass =
    variant === "hal" ? "msg-avatar msg-avatar--assistant msg-avatar--hal" : "msg-avatar msg-avatar--assistant";

  return (
    <div className={`${assistantClass} ${className}`.trim()} aria-hidden="true">
      <GroveeLogoMark size="xs" animated={false} />
    </div>
  );
}
