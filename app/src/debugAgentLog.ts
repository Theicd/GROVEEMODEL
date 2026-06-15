export const agentDebugLog = (
  hypothesisId: string,
  location: string,
  message: string,
  data: Record<string, unknown>,
  runId = "pre-fix",
): void => {
  if (typeof fetch === "undefined") return;
  // #region agent log
  fetch("http://127.0.0.1:7358/ingest/1cd0d1c8-df2a-45a6-9987-400b3ea6ac5a", { method: "POST", headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "f15741" }, body: JSON.stringify({ sessionId: "f15741", runId, hypothesisId, location, message, data, timestamp: Date.now() }) }).catch(() => {});
  // #endregion
};
