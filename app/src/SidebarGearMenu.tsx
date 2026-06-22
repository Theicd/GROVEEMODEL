import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { SidebarSettingsIcon } from "./SidebarSettingsIcon";

export type SidebarGearAction =
  | "settings"
  | "plugins"
  | "activity"
  | "presentation-qa"
  | "vision";

const MENU_ITEMS: {
  id: SidebarGearAction;
  icon: string;
  label: string;
  title?: string;
}[] = [
  { id: "settings", icon: "⚙", label: "הגדרות המודל", title: "Gemma · מנוע חישוב" },
  { id: "plugins", icon: "🧩", label: "מרכז התוספים", title: "חיפוש · API · RSS" },
  { id: "activity", icon: "📋", label: "פעילות המודל", title: "הנחיות ותשובות" },
  { id: "presentation-qa", icon: "✦", label: "בדיקת מצגת", title: "80 שאלות מצגת" },
  { id: "vision", icon: "👁", label: "Vision Inspector", title: "זיהוי חי במצלמה" },
];

type PopCoords = {
  bottom: number;
  left?: number;
  right?: number;
};

type Props = {
  variant: "rail" | "footer";
  onSelect: (action: SidebarGearAction) => void;
  showPresentationQa?: boolean;
  visionDisabled?: boolean;
  activityCount?: number;
};

function computeRailPopoverCoords(
  trigger: HTMLElement,
  popHeight: number,
): PopCoords {
  const rect = trigger.getBoundingClientRect();
  const gap = 10;
  const margin = 12;
  const towardCenterFromRight = rect.left + rect.width / 2 > window.innerWidth / 2;

  let bottom = window.innerHeight - rect.bottom;
  const topIfAnchoredBottom = rect.bottom - popHeight - gap;
  if (topIfAnchoredBottom < margin) {
    bottom = Math.max(margin, window.innerHeight - rect.top - popHeight - gap);
  }
  bottom = Math.min(bottom, window.innerHeight - margin);

  if (towardCenterFromRight) {
    return { right: window.innerWidth - rect.left + gap, bottom };
  }
  return { left: rect.right + gap, bottom };
}

export function SidebarGearMenu({
  variant,
  onSelect,
  showPresentationQa = true,
  visionDisabled = false,
  activityCount = 0,
}: Props) {
  const [open, setOpen] = useState(false);
  const [railCoords, setRailCoords] = useState<PopCoords | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);

  const items = MENU_ITEMS.filter((item) => item.id !== "presentation-qa" || showPresentationQa);

  const updateRailPosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const popHeight = popRef.current?.offsetHeight ?? items.length * 34 + 12;
    setRailCoords(computeRailPopoverCoords(trigger, popHeight));
  }, [items.length]);

  useLayoutEffect(() => {
    if (!open || variant !== "rail") {
      setRailCoords(null);
      return;
    }
    updateRailPosition();
    const pop = popRef.current;
    if (!pop || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => updateRailPosition());
    observer.observe(pop);
    return () => observer.disconnect();
  }, [open, variant, updateRailPosition, items.length]);

  useEffect(() => {
    if (!open || variant !== "rail") return;
    const onReflow = () => updateRailPosition();
    window.addEventListener("resize", onReflow);
    window.addEventListener("scroll", onReflow, true);
    return () => {
      window.removeEventListener("resize", onReflow);
      window.removeEventListener("scroll", onReflow, true);
    };
  }, [open, variant, updateRailPosition]);

  useEffect(() => {
    if (!open) return;
    const onPointer = (e: MouseEvent) => {
      const target = e.target as Node;
      if (rootRef.current?.contains(target)) return;
      if (popRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const pick = (action: SidebarGearAction) => {
    if (action === "vision" && visionDisabled) return;
    setOpen(false);
    onSelect(action);
  };

  const popover = open ? (
    <div
      ref={popRef}
      className={`sb-gear-menu-pop${variant === "rail" ? " sb-gear-menu-pop--portal" : ""}`}
      style={
        variant === "rail" && railCoords
          ? {
              bottom: railCoords.bottom,
              left: railCoords.left,
              right: railCoords.right,
            }
          : undefined
      }
      role="menu"
      aria-label="כלים והגדרות"
    >
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="menuitem"
          title={item.title}
          className={`sb-gear-menu-item${
            item.id === "vision" && visionDisabled ? " sb-gear-menu-item--disabled" : ""
          }`}
          disabled={item.id === "vision" && visionDisabled}
          onClick={() => pick(item.id)}
        >
          <span className="sb-gear-menu-item-icon" aria-hidden="true">
            {item.icon}
          </span>
          <span className="sb-gear-menu-item-label">{item.label}</span>
          {item.id === "activity" && activityCount > 0 ? (
            <span className="sb-gear-menu-badge">
              {activityCount > 99 ? "99+" : activityCount}
            </span>
          ) : null}
        </button>
      ))}
    </div>
  ) : null;

  return (
    <div
      ref={rootRef}
      className={`sb-gear-menu sb-gear-menu--${variant}${open ? " sb-gear-menu--open" : ""}`}
    >
      {variant === "footer" ? popover : null}
      <button
        ref={triggerRef}
        type="button"
        className={
          variant === "rail"
            ? "sb-rail-btn sb-rail-settings-foot sb-gear-menu-trigger"
            : "sb-settings-btn sb-gear-menu-trigger"
        }
        aria-label="תפריט כלים והגדרות"
        aria-expanded={open}
        aria-haspopup="menu"
        title="כלים והגדרות — Gemma, תוספים, פעילות ועוד"
        onClick={() => {
          setOpen((wasOpen) => {
            const next = !wasOpen;
            if (next && variant === "rail" && triggerRef.current) {
              setRailCoords(
                computeRailPopoverCoords(triggerRef.current, items.length * 34 + 12),
              );
            }
            return next;
          });
        }}
      >
        <SidebarSettingsIcon size={variant === "rail" ? 20 : 16} />
      </button>
      {variant === "rail" && popover && railCoords
        ? createPortal(popover, document.body)
        : null}
    </div>
  );
}
