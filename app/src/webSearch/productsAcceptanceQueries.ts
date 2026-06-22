import type { SearchIntent, SearchProviderId } from "./types";

export type ProductAcceptanceQuery = {
  id: string;
  query: string;
  expectIntents: SearchIntent[];
  expectProvider: SearchProviderId;
  /** Product name token expected in at least one hit title. */
  expectTitleIncludes?: string;
  /** Requires priceNis on top product hit (needs Cheapersal mock or live key). */
  expectPrice?: boolean;
  /** Requires imageUrl on top product hit. */
  expectImage?: boolean;
  notesHe?: string;
};

/** QA matrix for Israeli supermarket product + price pipeline. */
export const PRODUCT_ACCEPTANCE_QUERIES: ProductAcceptanceQuery[] = [
  {
    id: "P01",
    query: "כמה עולה חלב",
    expectIntents: ["products"],
    expectProvider: "israeli-products",
    expectTitleIncludes: "חלב",
    expectImage: true,
    expectPrice: true,
    notesHe: "שאלת מחיר בסיסית — קטלוג + Cheapersal",
  },
  {
    id: "P02",
    query: "כמה עולה לחם",
    expectIntents: ["products"],
    expectProvider: "israeli-products",
    expectTitleIncludes: "לחם",
    expectImage: false,
    notesHe: "לחם — לא כל הברקודים קיימים ב-Cheapersal",
  },
  {
    id: "P03",
    query: "כמה עולה קוטג",
    expectIntents: ["products"],
    expectProvider: "israeli-products",
    expectTitleIncludes: "קוטג",
    expectImage: true,
    expectPrice: true,
    notesHe: "מוצר מקרר שקיים ב-Cheapersal",
  },
  {
    id: "P04",
    query: "חלב תנובה",
    expectIntents: ["products"],
    expectProvider: "israeli-products",
    expectTitleIncludes: "תנובה",
    expectImage: true,
    notesHe: "חיפוש מוצר לפי מותג",
  },
  {
    id: "P05",
    query: "7290004131074",
    expectIntents: ["products"],
    expectProvider: "israeli-products",
    expectTitleIncludes: "חלב",
    expectImage: true,
    expectPrice: true,
    notesHe: "ברקוד ישראלי ישיר",
  },
  {
    id: "P06",
    query: "חלב",
    expectIntents: ["products"],
    expectProvider: "israeli-products",
    expectTitleIncludes: "חלב",
    expectImage: true,
    expectPrice: true,
    notesHe: "מילה בודדת — לשונית מוצרים",
  },
  {
    id: "P07",
    query: "לחם",
    expectIntents: ["products"],
    expectProvider: "israeli-products",
    expectTitleIncludes: "לחם",
    expectImage: true,
    notesHe: "לחם — מחיר ותמונה בלשונית מוצרים",
  },
];
