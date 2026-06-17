/**
 * App entry for Qwen summarizer — import from here in UI/bridge code.
 * Engine copy is wiped by sync:news; extensions live in engine-overlays/.
 */
export {
  armOnDemandDeepRead,
  bootSummarizer,
  expandQuery,
  getModelBootState,
  getPreferredSummarizerDevice,
  isSummarizerReady,
  setPreferredSummarizerDevice,
  subscribeModelBoot,
  summarizeArticle,
  translateHeadlineToEnglish,
  waitForSummarizer,
  parseSummarizerOutput,
} from "./engine/summarize/summarizerClient";

export type {
  ModelBootPhase,
  ModelBootState,
  SummarizerDevice,
  SummarizerResult,
  SummarizeArticleOptions,
} from "./engine/summarize/summarizerClient";
