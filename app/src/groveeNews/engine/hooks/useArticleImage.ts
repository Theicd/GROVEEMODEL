// @ts-nocheck
import { useEffect, useRef, useState } from "react";
import { enqueueArticleImageFetch } from "../media/imageFetchQueue";
import { canFetchArticlePageImage, resolveWarmArticleImage } from "../media/imageResolve";
import { hasRealImageUrl, isStockImageUrl, normalizeImageUrl } from "../media/imageFields";
import {
  detectStockProvider,
  searchStockImage,
  type StockImageProvider,
} from "../media/stockImageSearch";
import { isBlockedArticleHost } from "../feeds/blockedArticleHosts";
import { getRssItemByLink, patchArticleImage } from "../storage/db";

export type ArticleImageState = "idle" | "warm" | "loading" | "ready" | "unavailable" | "stock";

export type UseArticleImageOptions = {
  articleId: string;
  articleUrl: string;
  title?: string;
  /** Topic / lane keyword for stock-image fallback. */
  stockHint?: string;
  existing?: string;
  description?: string;
  enabled?: boolean;
  priority?: number;
  /** Try free stock libraries when article page has no hero image. */
  allowStockFallback?: boolean;
};

export type UseArticleImageResult = {
  src: string;
  state: ArticleImageState;
  stockProvider: StockImageProvider | null;
};

const PAGE_FETCH_TIMEOUT_MS = 12_000;
const REAL_IMAGE_RETRY_MS = 9_000;
const MAX_REAL_RETRIES = 10;

function applyRealImage(
  image: string,
  setSrc: (url: string) => void,
  setState: (state: ArticleImageState) => void,
  setStockProvider: (provider: StockImageProvider | null) => void,
  articleId: string,
  savedRef: { current: boolean },
): boolean {
  if (!hasRealImageUrl(image)) return false;
  setSrc(image);
  setState("ready");
  setStockProvider(null);
  if (articleId && !savedRef.current) {
    savedRef.current = true;
    void patchArticleImage(articleId, image);
  }
  return true;
}

export function useArticleImage({
  articleId,
  articleUrl,
  title = "",
  stockHint = "",
  existing = "",
  description = "",
  enabled = true,
  priority = 0,
  allowStockFallback = true,
}: UseArticleImageOptions): UseArticleImageResult {
  const normalizedExisting = normalizeImageUrl(existing);
  const realExisting = hasRealImageUrl(normalizedExisting) ? normalizedExisting : "";

  const [src, setSrc] = useState(realExisting);
  const [state, setState] = useState<ArticleImageState>(realExisting ? "ready" : "idle");
  const [stockProvider, setStockProvider] = useState<StockImageProvider | null>(null);
  const savedRef = useRef(false);

  useEffect(() => {
    savedRef.current = false;
    let cancelled = false;
    let settled = false;
    let retryTimer: number | undefined;
    let retryCount = 0;

    const warmInput = () => ({
      articleUrl,
      rssImage: realExisting,
      description,
    });

    const resolveRealFromSources = async (): Promise<string> => {
      const warm = resolveWarmArticleImage(warmInput());
      if (hasRealImageUrl(warm)) return warm;

      const rss = await getRssItemByLink(articleUrl);
      const rssWarm = resolveWarmArticleImage({
        articleUrl,
        rssImage: rss?.image,
        description: rss?.description ?? description,
      });
      if (hasRealImageUrl(rssWarm)) return rssWarm;

      if (!enabled || !articleUrl || isBlockedArticleHost(articleUrl)) return "";
      if (!canFetchArticlePageImage(articleUrl)) return "";

      const page = await enqueueArticleImageFetch(articleUrl, priority);
      return hasRealImageUrl(page) ? page : "";
    };

    const stopRetry = () => {
      if (retryTimer !== undefined) {
        window.clearInterval(retryTimer);
        retryTimer = undefined;
      }
    };

    const scheduleRealRetry = () => {
      if (retryTimer !== undefined || !articleUrl) return;
      retryTimer = window.setInterval(() => {
        if (cancelled || retryCount >= MAX_REAL_RETRIES) {
          stopRetry();
          return;
        }
        retryCount += 1;
        void resolveRealFromSources().then((image) => {
          if (cancelled || !hasRealImageUrl(image)) return;
          applyRealImage(image, setSrc, setState, setStockProvider, articleId, savedRef);
          stopRetry();
        });
      }, REAL_IMAGE_RETRY_MS);
    };

    const showStockPlaceholder = (): boolean => {
      if (!allowStockFallback || !title.trim() || !enabled) return false;
      setState("loading");
      void searchStockImage(title, stockHint).then((hit) => {
        if (cancelled) return;
        if (hit) {
          setSrc(hit.url);
          setState("stock");
          setStockProvider(hit.provider);
          scheduleRealRetry();
        } else {
          setSrc("");
          setState("unavailable");
          setStockProvider(null);
        }
      });
      return true;
    };

    if (realExisting) {
      setSrc(realExisting);
      setState("ready");
      setStockProvider(null);
      return;
    }

    const warm = resolveWarmArticleImage(warmInput());
    if (hasRealImageUrl(warm)) {
      applyRealImage(warm, setSrc, setState, setStockProvider, articleId, savedRef);
      return;
    }

    if (!enabled || !articleUrl || isBlockedArticleHost(articleUrl)) {
      setSrc("");
      setState("idle");
      setStockProvider(null);
      return () => {
        cancelled = true;
        stopRetry();
      };
    }

    if (!canFetchArticlePageImage(articleUrl)) {
      if (showStockPlaceholder()) {
        return () => {
          cancelled = true;
          stopRetry();
        };
      }
      setState("unavailable");
      setStockProvider(null);
      return () => {
        cancelled = true;
      };
    }

    setState("loading");
    setStockProvider(null);
    const timeout = window.setTimeout(() => {
      if (!cancelled && !settled) setState("unavailable");
    }, PAGE_FETCH_TIMEOUT_MS);

    void resolveRealFromSources().then((image) => {
      window.clearTimeout(timeout);
      if (cancelled) return;
      settled = true;

      if (hasRealImageUrl(image)) {
        applyRealImage(image, setSrc, setState, setStockProvider, articleId, savedRef);
        return;
      }

      if (showStockPlaceholder()) return;

      setSrc("");
      setState("unavailable");
      setStockProvider(null);
    });

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
      stopRetry();
    };
  }, [articleId, articleUrl, title, stockHint, existing, description, enabled, priority, allowStockFallback, realExisting]);

  const displaySrc = hasRealImageUrl(src)
    ? src
    : state === "stock" && src
      ? src
      : realExisting;

  const displayStock =
    stockProvider ?? (displaySrc && isStockImageUrl(displaySrc) ? detectStockProvider(displaySrc) : null);

  return { src: displaySrc, state, stockProvider: displayStock };
}
