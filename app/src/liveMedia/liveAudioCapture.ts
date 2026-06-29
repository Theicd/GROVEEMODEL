import {
  chunkHasSpeech,
  LIVE_CHUNK_OVERLAP_SECONDS,
  LIVE_CHUNK_SECONDS,
} from "./liveCaptionRoll";

export function resampleTo16kMono(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === 16000) return input;
  const outLen = Math.max(1, Math.floor((input.length * 16000) / inputRate));
  const out = new Float32Array(outLen);
  const step = inputRate / 16000;
  for (let i = 0; i < outLen; i += 1) {
    const pos = i * step;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const t = pos - i0;
    out[i] = input[i0] * (1 - t) + input[i1] * t;
  }
  return out;
}

export type TabAudioTap = {
  cleanup: () => void;
};

/** Capture tab audio silently via Web Audio — no double-playback. */
export async function startTabAudioTap(
  onChunk: (samples: Float32Array, sampleRate: number, audioStartMs: number) => void,
  onEnded?: () => void,
): Promise<TabAudioTap> {
  const constraints = {
    video: true,
    audio: true,
    preferCurrentTab: true,
    selfBrowserSurface: "include",
  } as MediaStreamConstraints;

  const displayStream = await navigator.mediaDevices.getDisplayMedia(constraints);
  for (const vt of displayStream.getVideoTracks()) {
    vt.enabled = false;
  }

  const audioTrack = displayStream.getAudioTracks()[0];
  if (!audioTrack) {
    displayStream.getTracks().forEach((t) => t.stop());
    throw new Error("no-audio");
  }

  const ctx = new AudioContext();
  await ctx.resume();
  const source = ctx.createMediaStreamSource(displayStream);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const nativeRate = ctx.sampleRate;
  let collected: number[] = [];
  let closed = false;

  const cleanup = () => {
    if (closed) return;
    closed = true;
    processor.onaudioprocess = null;
    processor.disconnect();
    source.disconnect();
    mute.disconnect();
    void ctx.close();
    displayStream.getTracks().forEach((t) => t.stop());
  };

  processor.onaudioprocess = (ev) => {
    if (closed) return;
    const ch = ev.inputBuffer.getChannelData(0);
    for (let i = 0; i < ch.length; i += 1) collected.push(ch[i]);
    const needed = Math.floor(nativeRate * LIVE_CHUNK_SECONDS);
    const overlapKeep = Math.floor(nativeRate * LIVE_CHUNK_OVERLAP_SECONDS);
    while (collected.length >= needed) {
      const slice = collected.splice(0, needed);
      const chunk = new Float32Array(slice);
      const chunkDurationMs = (needed / nativeRate) * 1000;
      const audioStartMs = Date.now() - chunkDurationMs;
      if (chunkHasSpeech(chunk)) onChunk(chunk, nativeRate, audioStartMs);
      if (overlapKeep > 0 && slice.length >= overlapKeep) {
        collected.unshift(...slice.slice(slice.length - overlapKeep));
      }
    }
  };

  source.connect(processor);
  processor.connect(mute);
  mute.connect(ctx.destination);

  audioTrack.addEventListener("ended", () => {
    cleanup();
    onEnded?.();
  });

  return { cleanup };
}
