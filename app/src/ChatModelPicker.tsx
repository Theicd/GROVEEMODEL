import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  GEMMA_RACK_ID,
  getRackModelById,
  modalityIcon,
  pickableRackModels,
  rackEntryTagLabel,
  setSelectedModelId,
  type ModelModality,
  type RackModelEntry,
} from "./modelRack/modelRack";
import {
  rackPickerBadge,
  rackPickerHint,
  rackPickerTitle,
  sortImageRackEntries,
} from "./modelRack/modelRackDisplay";

type Props = {
  rack: RackModelEntry[];
  selectedId: string;
  onSelect: (id: string) => void;
  disabled?: boolean;
};

const MODALITY_ORDER: ModelModality[] = ["text", "image", "code", "video", "audio", "vision"];

const MODALITY_LABELS: Record<ModelModality, string> = {
  text: "שיחה",
  image: "יצירת תמונה",
  code: "קוד",
  video: "וידאו",
  audio: "אודיו",
  vision: "ראייה",
};

function sortGroupItems(modality: ModelModality, items: RackModelEntry[]): RackModelEntry[] {
  if (modality === "image") return sortImageRackEntries(items);
  return items;
}

export function ChatModelPicker({ rack, selectedId, onSelect, disabled }: Props) {
  const pickable = pickableRackModels(rack);
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number; minWidth: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);

  const selected = getRackModelById(selectedId, pickable) ?? getRackModelById(GEMMA_RACK_ID, pickable)!;

  const updatePosition = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const margin = 8;
    const minWidth = Math.max(rect.width, 280);
    let left = rect.left;
    if (left + minWidth > window.innerWidth - margin) {
      left = window.innerWidth - minWidth - margin;
    }
    setCoords({ top: rect.bottom + 6, left: Math.max(margin, left), minWidth });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    updatePosition();
  }, [open, pickable.length, updatePosition]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (triggerRef.current?.contains(t) || popRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onScroll = () => updatePosition();
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onScroll);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onScroll);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [open, updatePosition]);

  const grouped = MODALITY_ORDER.map((modality) => ({
    modality,
    items: sortGroupItems(
      modality,
      pickable.filter((r) => r.modality === modality),
    ),
  })).filter((g) => g.items.length > 0);

  const pick = (id: string) => {
    setSelectedModelId(id);
    onSelect(id);
    setOpen(false);
  };

  const renderItem = (row: RackModelEntry) => {
    const isActive = row.id === selectedId;
    const badge = rackPickerBadge(row);
    const hint = rackPickerHint(row);
    const tag = rackEntryTagLabel(row);
    const isImage = row.modality === "image";

    return (
      <button
        key={row.id}
        type="button"
        role="option"
        aria-selected={isActive}
        className={`chat-model-picker-item${isActive ? " is-active" : ""}${isImage ? " chat-model-picker-item--image" : ""}`}
        onClick={() => pick(row.id)}
        title={hint ?? rackPickerTitle(row)}
      >
        {badge ? (
          <span
            className="chat-model-picker-badge"
            style={{ "--badge-accent": badge.accent } as React.CSSProperties}
            aria-hidden="true"
          >
            {badge.badge}
          </span>
        ) : (
          <span className="chat-model-picker-item-icon" aria-hidden="true">
            {modalityIcon(row.modality)}
          </span>
        )}
        <span className="chat-model-picker-item-body">
          <span className="chat-model-picker-item-name">{rackPickerTitle(row)}</span>
          {hint ? <span className="chat-model-picker-item-sub">{hint}</span> : null}
        </span>
        {tag ? <span className="chat-model-picker-item-tag">{tag}</span> : null}
      </button>
    );
  };

  const menu =
    open && coords
      ? createPortal(
          <div
            ref={popRef}
            className="chat-model-picker-pop"
            style={{ top: coords.top, left: coords.left, minWidth: coords.minWidth }}
            role="listbox"
            aria-label="בחירת מודל"
          >
            {grouped.map(({ modality, items }) => (
              <div key={modality} className="chat-model-picker-group">
                <div className="chat-model-picker-group-label">
                  {modalityIcon(modality)} {MODALITY_LABELS[modality]}
                </div>
                {modality === "image" ? (
                  <div className="chat-model-picker-image-grid">{items.map(renderItem)}</div>
                ) : (
                  items.map(renderItem)
                )}
              </div>
            ))}
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="chat-header-brand chat-header-brand--picker"
        dir="ltr"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`מודל נבחר: ${rackPickerTitle(selected)}`}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="chat-header-brand-icon" aria-hidden="true">
          {modalityIcon(selected.modality)}
        </span>
        <span className="chat-header-brand-name">{rackPickerTitle(selected)}</span>
        <svg
          className={`chat-header-brand-chevron${open ? " is-open" : ""}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          aria-hidden="true"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {menu}
    </>
  );
}
