/** GROVEE stub — VLM disabled; Gemma handles chat vision. */

export class VlmSceneDescriber {
  async init(_onProgress?: (msg: string) => void): Promise<void> {
    // no-op
  }

  async describe(_source: HTMLVideoElement, _force = false): Promise<string> {
    return "";
  }

  dispose(): void {
    // no-op
  }
}
