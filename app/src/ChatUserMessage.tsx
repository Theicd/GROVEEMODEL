import { useEffect, useRef } from "react";

type ChatUserMessageProps = {
  isEditing: boolean;
  editDraft: string;
  canEdit: boolean;
  isRtl: boolean;
  onStartEdit: () => void;
  onCancelEdit: () => void;
  onDraftChange: (value: string) => void;
  onSaveEdit: () => void;
  children: React.ReactNode;
};

export function ChatUserMessage({
  isEditing,
  editDraft,
  canEdit,
  isRtl,
  onStartEdit,
  onCancelEdit,
  onDraftChange,
  onSaveEdit,
  children,
}: ChatUserMessageProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!isEditing) return;
    const el = textareaRef.current;
    if (!el) return;
    el.focus({ preventScroll: true });
    el.setSelectionRange(el.value.length, el.value.length);
  }, [isEditing]);

  if (isEditing) {
    return (
      <div className="msg-user-edit" dir={isRtl ? "rtl" : "ltr"}>
        <textarea
          ref={textareaRef}
          className="msg-user-edit-input"
          value={editDraft}
          onChange={(e) => onDraftChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              onSaveEdit();
            }
            if (e.key === "Escape") {
              e.preventDefault();
              onCancelEdit();
            }
          }}
          rows={3}
          aria-label="עריכת הודעה"
        />
        <div className="msg-user-edit-actions">
          <button
            type="button"
            className="msg-user-edit-save"
            onClick={onSaveEdit}
            disabled={!editDraft.trim()}
          >
            שמור ושלח
          </button>
          <button type="button" className="msg-user-edit-cancel" onClick={onCancelEdit}>
            ביטול
          </button>
          <span className="msg-user-edit-hint">Ctrl+Enter לשליחה · Esc לביטול</span>
        </div>
      </div>
    );
  }

  return (
    <div className="msg-user-wrap">
      {children}
      {canEdit ? (
        <div className="msg-user-toolbar">
          <button type="button" className="msg-user-action" onClick={onStartEdit} title="ערוך הודעה">
            <span className="msg-user-action-icon" aria-hidden="true">
              ✎
            </span>
            ערוך
          </button>
        </div>
      ) : null}
    </div>
  );
}
