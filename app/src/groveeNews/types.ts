export type GroveeNewsCard = {
  id: string;
  title: string;
  titleOriginal: string;
  source: string;
  sourceKey: string;
  url: string;
  image: string;
  score: number;
  publishedTs: number;
  summary?: string;
  laneId?: string;
  laneLabel?: string;
  laneIcon?: string;
};

export type GroveeTopicCard = GroveeNewsCard & {
  laneId: string;
  laneLabel: string;
  laneIcon: string;
  query: string;
  matchLabel: "high" | "medium" | "low";
};

export type GroveeTopicsBundle = {
  generatedAt: number;
  cards: GroveeTopicCard[];
  stats: {
    totalLanes: number;
    lanesWithHits: number;
  };
};

export type NewsPanelMode = "search" | "topics";

export type NewsPanelPayload = {
  mode: NewsPanelMode;
  query: string;
  cards: GroveeNewsCard[];
  generatedAt: number;
};

export type ArticleReadResult = {
  title: string;
  summaryHe: string;
  /** English structured notes from Qwen — input for Gemma polish. */
  qwenDraft?: string;
  url: string;
  usedQwen: boolean;
  error?: string;
};

export type NewsSummaryGemmaProgress = {
  onGemmaToken?: (tokens: number) => void;
  onStreamChunk?: (text: string) => void;
};
