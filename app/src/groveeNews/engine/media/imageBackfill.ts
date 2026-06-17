// @ts-nocheck
import { enqueueArticleImageFetch } from "./imageFetchQueue";
import { canFetchArticlePageImage, resolveWarmArticleImage } from "./imageResolve";
import { isStockImageUrl } from "./imageFields";
import { searchStockImage } from "./stockImageSearch";
import { getRssItemByLink, listArticlesMissingImages, patchArticleImage } from "../storage/db";
import { yieldToMain } from "../util/yieldToMain";

async function resolveRealArticleImage(
  articleUrl: string,
  articleImage: string,
  articleText: string,
): Promise<string> {
  const rss = await getRssItemByLink(articleUrl);
  const warm = resolveWarmArticleImage({
    articleUrl,
    rssImage: rss?.image ?? articleImage,
    description: rss?.description ?? articleText,
  });
  if (warm && !isStockImageUrl(warm)) return warm;

  if (canFetchArticlePageImage(articleUrl)) {
    const page = await enqueueArticleImageFetch(articleUrl, -1);
    if (page && !isStockImageUrl(page)) return page;
  }

  return "";
}

/** Slowly fill missing images using warm sources first, then throttled page fetch. Stock is UI-only — never persisted. */
export async function backfillMissingImages(max = 6): Promise<number> {
  const missing = await listArticlesMissingImages(max);
  let filled = 0;

  for (let i = 0; i < missing.length; i++) {
    const article = missing[i];
    const image = await resolveRealArticleImage(
      article.url,
      article.image ?? "",
      article.articleText ?? "",
    );

    if (image) {
      await patchArticleImage(article.id, image);
      filled++;
    }

    if (i % 2 === 1) await yieldToMain();
  }

  return filled;
}
