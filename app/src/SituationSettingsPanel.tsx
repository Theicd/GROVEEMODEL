import { useState } from "react";
import {
  DEFAULT_SITUATION_RULES,
  loadSituationRegistry,
  resetSituationRegistry,
  saveSituationRegistry,
  type SituationRule,
  type SituationTier,
} from "./situationRegistry";
import type { VisionBehaviorSettings } from "./visionSettings";
import type { PerformanceMode } from "./vision-lab/core/types";

const TIER_LABELS: Record<SituationTier, string> = {
  instant: "מיידי (ללא LLM)",
  llm_boot: "LLM בהפעלה",
  llm_change: "LLM בשינוי",
};

export function SituationSettingsPanel({
  vision,
  onVisionChange,
}: {
  vision: VisionBehaviorSettings;
  onVisionChange: (partial: Partial<VisionBehaviorSettings>) => void;
}) {
  const [rules, setRules] = useState<SituationRule[]>(() => loadSituationRegistry());
  const [savedFlash, setSavedFlash] = useState(false);

  const updateRule = (id: string, patch: Partial<SituationRule>) => {
    setRules((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  };

  const persistRules = () => {
    saveSituationRegistry(rules);
    setSavedFlash(true);
    window.setTimeout(() => setSavedFlash(false), 2000);
  };

  const handleResetRules = () => {
    setRules(resetSituationRegistry());
  };

  const perfModes: PerformanceMode[] = ["lite", "balanced", "full"];

  return (
    <div className="situation-settings">
      <section className="settings-card">
        <h3 className="settings-card-title">
          <span className="settings-card-dot" aria-hidden="true" />
          מצלמה וראייה (HAL)
        </h3>
        <div className="situation-settings-toggles">
          <label className="situation-toggle">
            <input
              type="checkbox"
              checked={vision.useBootDeepSnapshot}
              onChange={(e) => onVisionChange({ useBootDeepSnapshot: e.target.checked })}
            />
            <span>פענוח תמונה ראשוני (Gemma) בהפעלת מצלמה</span>
          </label>
          <label className="situation-toggle">
            <input
              type="checkbox"
              checked={vision.useLlmDeepVision}
              onChange={(e) => onVisionChange({ useLlmDeepVision: e.target.checked })}
            />
            <span>ראייה עמוקה (Gemma) בשינויים משמעותיים</span>
          </label>
          <label className="situation-toggle">
            <input
              type="checkbox"
              checked={vision.useLlmProactiveUtterance}
              onChange={(e) => onVisionChange({ useLlmProactiveUtterance: e.target.checked })}
            />
            <span>ליטוש תגובות פרואקטיביות עם Gemma (עומס גבוה)</span>
          </label>
          <label className="situation-toggle">
            <input
              type="checkbox"
              checked={vision.showDetectionCards}
              onChange={(e) => onVisionChange({ showDetectionCards: e.target.checked })}
            />
            <span>הצג כרטיסיות זיהוי ב-Vision Inspector</span>
          </label>
          <label className="situation-toggle">
            <input
              type="checkbox"
              checked={vision.logVisionToActivity}
              onChange={(e) => onVisionChange({ logVisionToActivity: e.target.checked })}
            />
            <span>רישום זיהוי ביומן פעילות</span>
          </label>
        </div>
        <div className="situation-perf-row">
          <span className="settings-field-label">ביצועי Vision Lab</span>
          <div className="vision-inspector-presets">
            {perfModes.map((mode) => (
              <button
                key={mode}
                type="button"
                className={vision.performanceMode === mode ? "active" : ""}
                onClick={() => onVisionChange({ performanceMode: mode })}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="settings-card">
        <h3 className="settings-card-title">
          <span className="settings-card-dot" aria-hidden="true" />
          זיהוי אסיטואציות
        </h3>
        <p className="situation-settings-hint">
          כל שורה = טריגר שיכול לגרום ל-HAL להגיב. Cooldown מונע ספאם. tier &quot;מיידי&quot; לא קורא ל-Gemma.
        </p>
        <div className="situation-rules-table-wrap">
          <table className="situation-rules-table">
            <thead>
              <tr>
                <th>פעיל</th>
                <th>שם</th>
                <th>מקור</th>
                <th>Tier</th>
                <th>Cooldown</th>
                <th>פרואקטיבי</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((rule) => (
                <tr key={rule.id} className={rule.enabled ? "" : "situation-rule-off"}>
                  <td>
                    <input
                      type="checkbox"
                      checked={rule.enabled}
                      onChange={(e) => updateRule(rule.id, { enabled: e.target.checked })}
                      aria-label={`הפעל ${rule.labelHe}`}
                    />
                  </td>
                  <td>
                    <div className="situation-rule-label">{rule.labelHe}</div>
                    <div className="situation-rule-sub">{rule.match}</div>
                  </td>
                  <td>{rule.source}</td>
                  <td>
                    <select
                      value={rule.tier}
                      onChange={(e) => updateRule(rule.id, { tier: e.target.value as SituationTier })}
                    >
                      {(Object.keys(TIER_LABELS) as SituationTier[]).map((t) => (
                        <option key={t} value={t}>
                          {TIER_LABELS[t]}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <input
                      type="number"
                      min={1000}
                      max={600000}
                      step={1000}
                      value={rule.cooldownMs}
                      onChange={(e) => updateRule(rule.id, { cooldownMs: Number(e.target.value) })}
                      className="situation-cooldown-input"
                    />
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      checked={rule.proactive}
                      onChange={(e) => updateRule(rule.id, { proactive: e.target.checked })}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rules.map((rule) => (
          <label key={`utt-${rule.id}`} className="settings-field settings-field--full situation-utterance-field">
            <span className="settings-field-label">תגובה — {rule.labelHe}</span>
            <textarea
              rows={2}
              value={rule.utteranceHe}
              onChange={(e) => updateRule(rule.id, { utteranceHe: e.target.value })}
              dir="rtl"
            />
          </label>
        ))}
        <div className="situation-rules-actions">
          <button type="button" className="settings-btn-ghost" onClick={handleResetRules}>
            איפוס אסיטואציות
          </button>
          <button type="button" className="settings-btn-save" onClick={persistRules}>
            {savedFlash ? "נשמר ✓" : "שמור אסיטואציות"}
          </button>
        </div>
        <p className="situation-settings-foot">
          ברירת מחדל: {DEFAULT_SITUATION_RULES.length} אסיטואציות. שינויים נשמרים בנפרד מהגדרות Gemma.
        </p>
      </section>
    </div>
  );
}
