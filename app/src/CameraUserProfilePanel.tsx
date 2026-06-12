import { useCallback, useEffect, useMemo, useState } from "react";
import type { UserProfile } from "./cameraSession";
import type { HistorySearchHit } from "./cameraUserMemory";

type Props = {
  profile: UserProfile;
  rollingSummary: string;
  messageCount: number;
  searchHits: HistorySearchHit[];
  searchQuery: string;
  onSearchChange: (q: string) => void;
  onSaveProfile: (patch: Partial<UserProfile>) => void;
  disabled?: boolean;
};

export function CameraUserProfilePanel({
  profile,
  rollingSummary,
  messageCount,
  searchHits,
  searchQuery,
  onSearchChange,
  onSaveProfile,
  disabled,
}: Props) {
  const [name, setName] = useState(profile.name);
  const [hobbiesText, setHobbiesText] = useState(profile.hobbies.join(", "));
  const [notes, setNotes] = useState(profile.notes);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    setName(profile.name);
    setHobbiesText(profile.hobbies.join(", "));
    setNotes(profile.notes);
    setDirty(false);
  }, [profile.name, profile.hobbies, profile.notes]);

  const save = useCallback(() => {
    const hobbies = hobbiesText
      .split(/[,،;|]/)
      .map((h) => h.trim())
      .filter(Boolean);
    onSaveProfile({ name, hobbies, notes });
    setDirty(false);
  }, [name, hobbiesText, notes, onSaveProfile]);

  const summaryLines = useMemo(
    () => rollingSummary.split("\n").filter(Boolean).slice(-4),
    [rollingSummary],
  );

  return (
    <div className="camera-profile-panel" dir="rtl">
      <p className="camera-sidebar-title">זיכרון HAL · פרופיל</p>
      <p className="camera-sidebar-meta">{messageCount} הודעות · נפרד מצ&apos;אט טקסט</p>

      <label className="camera-profile-field">
        <span>שם</span>
        <input
          type="text"
          value={name}
          disabled={disabled}
          placeholder="איך לקרוא לך?"
          onChange={(e) => {
            setName(e.target.value);
            setDirty(true);
          }}
        />
      </label>

      <label className="camera-profile-field">
        <span>תחביבים / עניין</span>
        <input
          type="text"
          value={hobbiesText}
          disabled={disabled}
          placeholder="מדע בדיוני, קונספירציות…"
          onChange={(e) => {
            setHobbiesText(e.target.value);
            setDirty(true);
          }}
        />
      </label>

      <label className="camera-profile-field">
        <span>הערות</span>
        <textarea
          value={notes}
          disabled={disabled}
          rows={2}
          placeholder="מה חשוב לזכור עליך"
          onChange={(e) => {
            setNotes(e.target.value);
            setDirty(true);
          }}
        />
      </label>

      {dirty ? (
        <button type="button" className="camera-profile-save" disabled={disabled} onClick={save}>
          שמור פרופיל
        </button>
      ) : null}

      {profile.hobbies.length && !dirty ? (
        <div className="camera-sidebar-topics">
          {profile.hobbies.map((h) => (
            <span key={h} className="camera-topic-chip">
              {h}
            </span>
          ))}
        </div>
      ) : null}

      {summaryLines.length ? (
        <div className="camera-summary-block">
          <p className="camera-summary-label">סיכום שיחה אחרונה</p>
          {summaryLines.map((line, i) => (
            <p key={i} className="camera-summary-line">
              {line}
            </p>
          ))}
        </div>
      ) : null}

      <label className="camera-profile-field camera-profile-search">
        <span>חיפוש בהיסטוריה</span>
        <input
          type="search"
          value={searchQuery}
          disabled={disabled}
          placeholder="מדע בדיוני, משחק…"
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </label>

      {searchQuery.trim() && searchHits.length ? (
        <ul className="camera-search-hits">
          {searchHits.map((h) => (
            <li key={h.message.id} className="camera-search-hit">
              <span className="camera-search-hit-role">
                {h.message.role === "user" ? "אתה" : "HAL"}
              </span>
              <span className="camera-search-hit-text">{h.snippet}</span>
            </li>
          ))}
        </ul>
      ) : searchQuery.trim().length >= 2 ? (
        <p className="camera-sidebar-note">אין תוצאות</p>
      ) : null}
    </div>
  );
}
