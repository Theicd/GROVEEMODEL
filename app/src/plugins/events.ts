export const PLUGIN_STATUS_EVENT = "grovee-plugin-status";

export const dispatchPluginStatusEvent = (): void => {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(PLUGIN_STATUS_EVENT));
};
