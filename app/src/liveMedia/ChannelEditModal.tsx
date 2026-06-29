import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "../searchResults/types";
import {
  USER_CHANNEL_CATEGORIES,
  VIEW_LANGUAGE_OPTIONS,
  categoryLabelEn,
  categoryLabelHe,
  normalizeChannelImageUrl,
  normalizeChannelStreamUrl,
  type ChannelUserOverride,
  type UserChannelCategory,
  type ViewLanguageCode,
} from "./channelUserTaxonomy";
import { hitChannelId } from "./channelDisplay";
import "./channelEditModal.css";

type Props = {
  hit: UnifiedSearchHit;
  channelName: string;
  catalogLogo?: string;
  catalogStream?: string;
  defaultBroadcastLanguage: ViewLanguageCode;
  initial: ChannelUserOverride;
  uiLang: ChatUiLanguage;
  onSave: (patch: ChannelUserOverride | null) => void | Promise<void>;
  onClose: () => void;
};

export function ChannelEditModal({
  hit,
  channelName,
  catalogLogo = "",
  catalogStream = "",
  defaultBroadcastLanguage,
  initial,
  uiLang,
  onSave,
  onClose,
}: Props) {
  const rtl = uiLang === "he";
  const nameInputRef = useRef<HTMLInputElement>(null);
  const [displayName, setDisplayName] = useState(initial.displayName ?? hit.title ?? channelName);
  const catalogStreamUrl = useMemo(
    () => normalizeChannelStreamUrl(catalogStream) || normalizeChannelStreamUrl(hit.mediaPlayUrl) || normalizeChannelStreamUrl(hit.url) || "",
    [catalogStream, hit.mediaPlayUrl, hit.url],
  );
  const [streamUrl, setStreamUrl] = useState(
    initial.streamUrl ?? hit.mediaPlayUrl ?? hit.url ?? catalogStreamUrl,
  );
  const [imageUrl, setImageUrl] = useState(initial.imageUrl ?? hit.imageUrl ?? "");
  const [imagePreviewOk, setImagePreviewOk] = useState(true);
  const [category, setCategory] = useState<UserChannelCategory>(
    initial.category ?? (hit.meta?.userCategory as UserChannelCategory) ?? "general",
  );
  const [broadcastLanguage, setBroadcastLanguage] = useState<ViewLanguageCode>(
    initial.broadcastLanguage ??
      (hit.meta?.broadcastLanguage as ViewLanguageCode | undefined) ??
      defaultBroadcastLanguage,
  );
  const [busy, setBusy] = useState(false);

  const L =
    uiLang === "he"
      ? {
          title: "עריכת ערוץ",
          name: "שם",
          stream: "קישור שידור (URL)",
          image: "תמונה (URL)",
          category: "קטגוריה",
          language: "שפה",
          save: "שמור",
          cancel: "ביטול",
          reset: "איפוס",
          clearName: "איפוס שם",
          clearStream: "איפוס קישור",
          clearImage: "הסר",
          imageInvalid: "קישור לא תקין",
          streamInvalid: "קישור שידור לא תקין",
        }
      : {
          title: "Edit channel",
          name: "Name",
          stream: "Stream link (URL)",
          image: "Image (URL)",
          category: "Category",
          language: "Language",
          save: "Save",
          cancel: "Cancel",
          reset: "Reset",
          clearName: "Reset name",
          clearStream: "Reset link",
          clearImage: "Clear",
          imageInvalid: "Invalid URL",
          streamInvalid: "Invalid stream URL",
        };

  const normalizedImage = useMemo(() => normalizeChannelImageUrl(imageUrl), [imageUrl]);
  const normalizedStream = useMemo(() => normalizeChannelStreamUrl(streamUrl), [streamUrl]);
  const catalogImage = useMemo(() => normalizeChannelImageUrl(catalogLogo), [catalogLogo]);
  const previewSrc = normalizedImage || catalogImage || "";

  useEffect(() => {
    nameInputRef.current?.focus();
    nameInputRef.current?.select();
  }, []);

  useEffect(() => {
    setImagePreviewOk(true);
  }, [previewSrc]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const imageUrlInvalid = imageUrl.trim().length > 0 && !normalizedImage;
  const streamUrlInvalid = streamUrl.trim().length > 0 && !normalizedStream;

  const handleSave = async () => {
    if (imageUrlInvalid || streamUrlInvalid) return;
    setBusy(true);
    try {
      const trimmed = displayName.trim();
      const img = normalizedImage;
      const stream = normalizedStream;
      await onSave({
        category,
        broadcastLanguage: broadcastLanguage !== defaultBroadcastLanguage ? broadcastLanguage : undefined,
        displayName: trimmed && trimmed !== channelName ? trimmed : undefined,
        imageUrl: img && img !== catalogImage ? img : undefined,
        streamUrl: stream && stream !== catalogStreamUrl ? stream : undefined,
      });
      onClose();
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    setBusy(true);
    try {
      await onSave(null);
      onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="lm-channel-edit-backdrop" role="presentation" onClick={onClose}>
      <div
        className="lm-channel-edit-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="lm-channel-edit-title"
        dir={rtl ? "rtl" : "ltr"}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="lm-channel-edit-head">
          <h2 id="lm-channel-edit-title" className="lm-channel-edit-title">
            {L.title}
          </h2>
          <p className="lm-channel-edit-original" title={channelName}>
            {channelName}
          </p>
        </header>

        <div className="lm-channel-edit-body">
          <label className="lm-channel-edit-field lm-channel-edit-field--compact">
            <span>{L.name}</span>
            <div className="lm-channel-edit-name-row">
              <input
                ref={nameInputRef}
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder={channelName}
                dir="auto"
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className="lm-channel-edit-name-btn"
                title={L.clearName}
                aria-label={L.clearName}
                onClick={() => setDisplayName(channelName)}
              >
                ×
              </button>
            </div>
          </label>

          <label className="lm-channel-edit-field lm-channel-edit-field--compact">
            <span>{L.stream}</span>
            <div className="lm-channel-edit-name-row">
              <input
                type="url"
                className="lm-channel-edit-stream-input"
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                placeholder="https://…/stream.m3u8"
                dir="ltr"
                autoComplete="off"
                spellCheck={false}
                inputMode="url"
              />
              <button
                type="button"
                className="lm-channel-edit-name-btn"
                title={L.clearStream}
                aria-label={L.clearStream}
                onClick={() => setStreamUrl(catalogStreamUrl)}
              >
                ×
              </button>
            </div>
            {streamUrlInvalid ? <small className="lm-channel-edit-warn">{L.streamInvalid}</small> : null}
          </label>

          <label className="lm-channel-edit-field lm-channel-edit-field--compact">
            <span>{L.image}</span>
            <div className="lm-channel-edit-image-row">
              {previewSrc ? (
                <span className="lm-channel-edit-thumb" aria-hidden="true">
                  {imagePreviewOk ? (
                    <img
                      src={previewSrc}
                      alt=""
                      referrerPolicy="no-referrer"
                      onError={() => setImagePreviewOk(false)}
                    />
                  ) : (
                    <span className="lm-channel-edit-thumb-fail">?</span>
                  )}
                </span>
              ) : null}
              <div className="lm-channel-edit-name-row lm-channel-edit-name-row--grow">
                <input
                  type="url"
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  placeholder="https://…"
                  dir="ltr"
                  autoComplete="off"
                  spellCheck={false}
                  inputMode="url"
                />
                <button
                  type="button"
                  className="lm-channel-edit-name-btn"
                  title={L.clearImage}
                  aria-label={L.clearImage}
                  onClick={() => setImageUrl("")}
                >
                  ×
                </button>
              </div>
            </div>
            {imageUrlInvalid ? <small className="lm-channel-edit-warn">{L.imageInvalid}</small> : null}
          </label>

          <div className="lm-channel-edit-row2">
            <label className="lm-channel-edit-field lm-channel-edit-field--compact">
              <span>{L.category}</span>
              <select value={category} onChange={(e) => setCategory(e.target.value as UserChannelCategory)}>
                {USER_CHANNEL_CATEGORIES.map((c) => (
                  <option key={c.id} value={c.id}>
                    {rtl ? categoryLabelHe(c.id) : categoryLabelEn(c.id)}
                  </option>
                ))}
              </select>
            </label>

            <label className="lm-channel-edit-field lm-channel-edit-field--compact">
              <span>{L.language}</span>
              <select
                value={broadcastLanguage}
                onChange={(e) => setBroadcastLanguage(e.target.value as ViewLanguageCode)}
              >
                {VIEW_LANGUAGE_OPTIONS.map((opt) => (
                  <option key={opt.code} value={opt.code}>
                    {rtl ? opt.nameHe : opt.nameEn}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <div className="lm-channel-edit-actions">
          <button type="button" className="lm-panel-btn" onClick={onClose} disabled={busy}>
            {L.cancel}
          </button>
          <button type="button" className="lm-panel-btn" onClick={() => void handleReset()} disabled={busy}>
            {L.reset}
          </button>
          <button
            type="button"
            className="lm-panel-btn lm-panel-btn--primary"
            onClick={() => void handleSave()}
            disabled={busy || imageUrlInvalid || streamUrlInvalid}
          >
            {L.save}
          </button>
        </div>
      </div>
    </div>
  );
}
