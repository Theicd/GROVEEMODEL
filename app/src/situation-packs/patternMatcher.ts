/** Match SituationPack trigger bundles against multi-signal context. */

import type { PackTriggers, SituationPack } from "./types";
import type { MatchContext } from "./patternContext";
import { signalCount, signalDurationSec, countInteractionSignals } from "./signalHistory";

const norm = (s: string) => s.trim().toLowerCase().replace(/\s+/g, "_");

const anyIn = (needles: string[], hay: string[]): boolean => {
  const set = new Set(hay.map(norm));
  return needles.some((n) => {
    const x = norm(n);
    return set.has(x) || [...set].some((h) => h.includes(x) || x.includes(h));
  });
};

const matchMotion = (level: "low" | "high" | "variable", ctx: MatchContext): boolean => {
  if (level === "low") return ctx.motionLevel < 0.12;
  if (level === "high") return ctx.motionLevel >= 0.45;
  return ctx.motionLevel >= 0.2 && ctx.motionLevel <= 0.75;
};

const matchHands = (mode: "inactive" | "active", ctx: MatchContext): boolean => {
  const active =
    ctx.obs.raisedHand ||
    ctx.obs.waving ||
    ctx.obs.pointing ||
    ctx.gestures.length > 0 ||
    ctx.motionLevel >= 0.25;
  return mode === "active" ? active : !active;
};

const primarySignalKey = (triggers: PackTriggers): string | null => {
  if (triggers.gestures?.length) return `gesture:${norm(triggers.gestures[0])}`;
  if (triggers.bodyLanguage?.length) return `body:${norm(triggers.bodyLanguage[0])}`;
  if (triggers.events?.length) return `event:${norm(triggers.events[0])}`;
  if (triggers.objects?.length) return `object:${norm(triggers.objects[0])}`;
  return null;
};

export const matchTriggers = (
  triggers: PackTriggers,
  ctx: MatchContext,
): { match: boolean; score: number } => {
  if (triggers.all?.length) {
    const subs = triggers.all.map((t) => matchTriggers(t, ctx));
    if (!subs.every((s) => s.match)) return { match: false, score: 0 };
    return { match: true, score: subs.reduce((a, s) => a + s.score, 0) / subs.length };
  }
  if (triggers.any?.length) {
    const subs = triggers.any.map((t) => matchTriggers(t, ctx));
    const hit = subs.filter((s) => s.match);
    if (!hit.length) return { match: false, score: 0 };
    return { match: true, score: Math.max(...hit.map((s) => s.score)) };
  }

  let checks = 0;
  let hits = 0;

  if (triggers.gestures?.length) {
    checks++;
    if (anyIn(triggers.gestures, ctx.gestures)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.bodyLanguage?.length) {
    checks++;
    if (anyIn(triggers.bodyLanguage, ctx.bodyLanguage)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.posture?.length) {
    checks++;
    if (triggers.posture.includes(ctx.human.posture)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.attention?.length) {
    checks++;
    if (triggers.attention.includes(ctx.human.attention)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.events?.length) {
    checks++;
    if (anyIn(triggers.events, ctx.events)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.objects?.length) {
    checks++;
    const objs = [
      ...ctx.objects,
      ctx.obs.holdingCup ? "cup" : "",
      ctx.snapshot.room.hasCup ? "cup" : "",
      ctx.snapshot.room.hasPhone ? "phone" : "",
      ctx.snapshot.room.hasLaptop ? "laptop" : "",
      ...ctx.snapshot.room.stableObjects.map(norm),
    ].filter(Boolean);
    if (anyIn(triggers.objects, objs)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.poseActions?.length) {
    checks++;
    if (anyIn(triggers.poseActions, ctx.poseActions)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.situations?.length) {
    checks++;
    if (triggers.situations.includes(ctx.situation.primary)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.motion) {
    checks++;
    if (matchMotion(triggers.motion, ctx)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.hands) {
    checks++;
    if (matchHands(triggers.hands, ctx)) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.minBodyScore) {
    for (const [k, min] of Object.entries(triggers.minBodyScore)) {
      checks++;
      const val = ctx.body[k as keyof typeof ctx.body];
      if (typeof val === "number" && val >= (min ?? 0)) hits++;
      else return { match: false, score: 0 };
    }
  }
  if (triggers.minEngagement !== undefined) {
    checks++;
    if (ctx.human.engagement >= triggers.minEngagement) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.maxEngagement !== undefined) {
    checks++;
    if (ctx.human.engagement <= triggers.maxEngagement) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.personPresent !== undefined) {
    checks++;
    if (ctx.obs.personPresent === triggers.personPresent) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.minSilenceSec !== undefined) {
    checks++;
    if (ctx.silenceSec >= triggers.minSilenceSec) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.minDurationSec !== undefined) {
    checks++;
    const key = primarySignalKey(triggers);
    let dur = key ? signalDurationSec(ctx.history, key, ctx.now) : 0;
    if (!dur && ctx.obs.personPresent) {
      dur = Math.max(
        signalDurationSec(ctx.history, "presence:person", ctx.now),
        ctx.snapshot.session.sceneAgeSec,
      );
    }
    if (dur >= triggers.minDurationSec) hits++;
    else return { match: false, score: 0 };
  }
  if (triggers.minRepetition !== undefined) {
    checks++;
    const window = triggers.timeWindowSec ?? 5;
    const key = primarySignalKey(triggers);
    const count = key
      ? signalCount(ctx.history, key, window, ctx.now)
      : countInteractionSignals(ctx.history, window, ctx.now);
    if (count >= triggers.minRepetition) hits++;
    else return { match: false, score: 0 };
  }

  if (checks === 0) return { match: false, score: 0 };
  const score = hits / checks;
  return { match: true, score: Math.max(0.35, score) };
};

export const matchSituationPacks = (
  packs: SituationPack[],
  ctx: MatchContext,
): Array<{ pack: SituationPack; score: number; confidence: number }> => {
  const out: Array<{ pack: SituationPack; score: number; confidence: number }> = [];
  for (const pack of packs) {
    if (!pack.enabled || !pack.proactive) continue;
    const { match, score } = matchTriggers(pack.triggers, ctx);
    if (!match) continue;
    const confidence = Math.min(1, score * (ctx.situation.confidence || 0.5) + 0.25);
    out.push({ pack, score, confidence });
  }
  out.sort((a, b) => {
    const pr =
      (b.pack.priority === "critical" ? 4 : b.pack.priority === "high" ? 3 : b.pack.priority === "medium" ? 2 : 1) -
      (a.pack.priority === "critical" ? 4 : a.pack.priority === "high" ? 3 : a.pack.priority === "medium" ? 2 : 1);
    if (pr !== 0) return pr;
    return b.score * b.confidence - a.score * a.confidence;
  });
  return out;
};
