/**
 * Match VisionResult signals against SituationRegistry with per-rule cooldowns.
 */

import type { VisionResult } from "./vision-lab/core/types";
import { makeSemanticEvent, normalizeLabel, type SemanticEvent } from "./worldMemory";
import type { SituationRule } from "./situationRegistry";

const norm = (s: string) => s.trim().toLowerCase();

export type SituationTriggerState = {
  lastFiredAt: Map<string, number>;
};

export const createSituationTriggerState = (): SituationTriggerState => ({
  lastFiredAt: new Map(),
});

const canFire = (state: SituationTriggerState, rule: SituationRule): boolean => {
  const last = state.lastFiredAt.get(rule.id) ?? 0;
  if (Date.now() - last < rule.cooldownMs) return false;
  state.lastFiredAt.set(rule.id, Date.now());
  return true;
};

const collectSignals = (result: VisionResult): Array<{ source: SituationRule["source"]; name: string }> => {
  const out: Array<{ source: SituationRule["source"]; name: string }> = [];

  for (const g of result.staticGestures) {
    out.push({ source: "static_gesture", name: norm(g.name) });
  }
  for (const g of result.motionGestures) {
    out.push({ source: "motion_gesture", name: norm(g.name) });
  }
  for (const a of result.poseActions) {
    out.push({ source: "pose_action", name: norm(a.name) });
  }
  for (const c of result.bodyLanguage) {
    out.push({ source: "body_language", name: norm(c.signal) });
  }
  for (const i of result.interactions) {
    out.push({ source: "interaction", name: norm(i.name) });
  }
  for (const e of result.events) {
    out.push({ source: "event", name: norm(e.name) });
  }
  for (const o of result.objects) {
    out.push({ source: "object", name: norm(o.displayLabel || o.label) });
  }
  for (const f of result.fingerStates) {
    out.push({ source: "finger_count", name: String(f.count) });
  }

  return out;
};

const matchesRule = (rule: SituationRule, signal: { source: SituationRule["source"]; name: string }): boolean => {
  if (rule.source !== signal.source) return false;
  const m = norm(rule.match);
  return signal.name === m || signal.name.includes(m) || m.includes(signal.name);
};

export const evaluateSituationTriggers = (
  result: VisionResult,
  rules: SituationRule[],
  state: SituationTriggerState,
): SemanticEvent[] => {
  const signals = collectSignals(result);
  const events: SemanticEvent[] = [];
  const firedSubjects = new Set<string>();

  for (const rule of rules) {
    if (!rule.enabled || !rule.proactive) continue;
    if (firedSubjects.has(rule.subject)) continue;

    const hit = signals.some((s) => matchesRule(rule, s));
    if (!hit) continue;
    if (!canFire(state, rule)) continue;

    firedSubjects.add(rule.subject);
    events.push(
      makeSemanticEvent(
        "activity_change",
        rule.label,
        rule.subject,
        rule.tier === "instant",
      ),
    );
  }

  return events;
};

/** Map lab event name to registry subject when EventRuleEngine already fired. */
export const registrySubjectFromLabEvent = (eventName: string, rules: SituationRule[]): string | null => {
  const n = norm(eventName);
  for (const rule of rules) {
    if (!rule.enabled) continue;
    if (rule.source === "event" && (n === norm(rule.match) || n.includes(norm(rule.match)))) {
      return rule.subject;
    }
  }
  if (/calling for attention|greeting/i.test(n)) return "wave";
  if (/phone usage/i.test(n)) return "focused_work";
  if (/drinking|holding cup/i.test(n)) return "stood_with_drink";
  if (/like|thumbs up/i.test(n)) return "gesture:thumbs_up";
  return `lab:${normalizeLabel(eventName)}`;
};
