# Premium Vision QA (Face + Emotion + Always-On Models)

Automated: `npm run dev` then `npm run qa:vision`  
Unit policy: `npm test` includes `vision-always-on.test.ts`

## Boot flow (camera on)

| Step | What happens | Expect | Auto |
|------|----------------|--------|------|
| B1 | Camera toggled on | Vision Lab loads YOLO, pose, hands, face-api | qa:vision #3 |
| B2 | First world sync | Sensor baseline captured | manual |
| B3 | Boot deep snapshot (if enabled) | **One** Gemma 4 scene pass; YOLO/face/hands **keep running** | qa:vision #7 |
| B4 | After boot | Continuous YOLO + face/emotion on schedule (intervals in inspector — do not change defaults in QA) | qa:vision #6 |

## Face detection (critical)

| # | Check | Pass criteria |
|---|--------|----------------|
| F1 | Face module status | `ready` or `scanning` — never stuck on `loading`/`error` |
| F2 | Face count | ≥1 when face in frame |
| F3 | Age estimate | Integer > 0 on Face card |
| F4 | Gender estimate | Male/Female on Face card |
| F5 | Gaze | Left/Center/Right shown |
| F6 | Video overlay | Yellow bbox + Face # label on inspector feed |
| F7 | `window.__groveeVisionProbe` (dev) | `faceOk === true` within 90s |

## Emotion detection (critical)

| # | Check | Pass criteria |
|---|--------|----------------|
| E1 | Dominant emotion | Non-empty label + score > 0% |
| E2 | Emotion meter strip | 6 bars (happy, neutral, sad, angry, surprised, fearful) under video |
| E3 | Live animation | Bar widths change when expression changes (transition ~150ms) |
| E4 | Probe | `emotionOk === true` within 90s |
| E5 | Disclaimer | “Estimate only — not clinical.” visible in full dashboard |

## Always-on models (no auto-pause)

| # | Scenario | Expect |
|---|----------|--------|
| A1 | User sends chat while camera on | YOLO objects still update; `pipelinePaused === false` |
| A2 | Gemma boot deep snapshot | Status shows Gemma reason; YOLO FPS > 0; objects still detected |
| A3 | Background `analyze_scene` | Hands/pose/face/YOLO continue |
| A4 | Tab hidden (optional) | Pipeline **not** stopped (models keep running) |
| A5 | Unit test | No `heavyPaused` / `setHeavyPaused` in pipeline or runner |

## Model interval settings

**Do not modify** sample intervals during QA unless testing a specific preset bug:
- Y, P, H, F, E, UI chips in Vision Inspector toolbar
- balanced / lite / full presets

## Manual-only checks

| # | Steps | Expect |
|---|--------|--------|
| M1 | Wave / fingers / pose | Cards update; proactive speech optional via settings |
| M2 | World Memory panel | Finger counts, pose, objects match cards |
| M3 | Activity log | YOLO + vision lines while camera runs |
| M4 | Low-tier device | Face + emotion toggles **not** forced off |

## Failure triage

1. `npm run models:face` — 4 shards in `public/models/face-api/`
2. DevTools console — `[VisionPipeline] face/emotion` warnings
3. Inspector Face card message — model load error text
4. `tests/qa-vision-premium-results.json` after automated run
