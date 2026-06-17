// @ts-nocheck
/** Keep search interaction active when focus moves to the submit button (avoids feed remount). */
export function shouldRetainSearchFocus(
  relatedTarget: EventTarget | null,
  form: HTMLFormElement | null,
): boolean {
  if (!relatedTarget || !form) return false;
  if (typeof form.contains !== "function") return false;
  return form.contains(relatedTarget as Node);
}
