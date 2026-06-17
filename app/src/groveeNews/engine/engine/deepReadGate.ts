// @ts-nocheck
import { isAiDeepReadEnabled } from "../settings/aiMode";

/** When false, pipeline skips fetch+extract+Qwen batch after RSS poll. */
export function isDeepReadEnabled(): boolean {
  return isAiDeepReadEnabled();
}
