import { HF_INFERENCE_CHAT_URL } from "./hfModelTypes";

export function buildHfCurlSnippet(modelId: string, endpoint = HF_INFERENCE_CHAT_URL): string {
  const body = JSON.stringify({
    model: modelId,
    messages: [{ role: "user", content: "Hello" }],
    max_tokens: 64,
  });
  return [
    `curl ${endpoint} \\`,
    '  -H "Authorization: Bearer $HF_TOKEN" \\',
    '  -H "Content-Type: application/json" \\',
    `  -d '${body}'`,
  ].join("\n");
}

export function buildHfPythonSnippet(modelId: string, endpoint = HF_INFERENCE_CHAT_URL): string {
  return [
    "import os",
    "import requests",
    "",
    `API_URL = "${endpoint}"`,
    'headers = {"Authorization": f"Bearer {os.environ[\'HF_TOKEN\']}"}',
    "payload = {",
    `    "model": "${modelId}",`,
    '    "messages": [{"role": "user", "content": "Hello"}],',
    '    "max_tokens": 64,',
    "}",
    "response = requests.post(API_URL, headers=headers, json=payload, timeout=60)",
    "response.raise_for_status()",
    "print(response.json())",
  ].join("\n");
}

export function statusBadgeLabel(status: string, uiLang: "he" | "en"): string {
  const s = status.toUpperCase();
  if (uiLang === "he") {
    if (s === "WORKING") return "עובד ב-API";
    if (s === "PROVIDER REQUIRED") return "דורש HF Token";
    if (s === "RATE LIMITED") return "מוגבל בקצב";
    if (s === "LOADING") return "נטען…";
    if (s === "MODEL NOT SUPPORTED") return "לא נתמך ב-Chat API";
    return "לא נבדק / שגיאה";
  }
  if (s === "WORKING") return "API working";
  if (s === "PROVIDER REQUIRED") return "HF token required";
  if (s === "RATE LIMITED") return "Rate limited";
  if (s === "LOADING") return "Loading…";
  if (s === "MODEL NOT SUPPORTED") return "Not a chat model";
  return "Not probed / error";
}

export function accessModeLabel(mode: string, uiLang: "he" | "en"): string {
  if (mode === "FREE") return uiLang === "he" ? "חינם (חשבון HF)" : "Free (HF account)";
  if (mode === "TOKEN") return uiLang === "he" ? "עם Token" : "With token";
  return uiLang === "he" ? "לא ידוע" : "Unknown";
}
