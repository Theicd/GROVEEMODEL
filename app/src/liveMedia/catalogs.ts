export type LiveMediaCategory = {
  id: string;
  name: string;
  nameHe: string;
};

export const LIVE_MEDIA_CATEGORIES: LiveMediaCategory[] = [
  { id: "news", name: "News", nameHe: "חדשות" },
  { id: "sports", name: "Sports", nameHe: "ספורט" },
  { id: "music", name: "Music", nameHe: "מוזיקה" },
  { id: "entertainment", name: "Entertainment", nameHe: "בידור" },
  { id: "comedy", name: "Comedy", nameHe: "קומדיה" },
  { id: "movies", name: "Movies", nameHe: "סרטים" },
  { id: "kids", name: "Kids", nameHe: "ילדים" },
  { id: "documentary", name: "Documentary", nameHe: "תיעוד" },
  { id: "general", name: "General", nameHe: "כללי" },
  { id: "culture", name: "Culture", nameHe: "תרבות" },
];

export const LIVE_MEDIA_COUNTRIES: { code: string; name: string; nameHe: string; flag: string }[] = [
  { code: "il", name: "Israel", nameHe: "ישראל", flag: "🇮🇱" },
  { code: "us", name: "United States", nameHe: "ארה״ב", flag: "🇺🇸" },
  { code: "gb", name: "United Kingdom", nameHe: "בריטניה", flag: "🇬🇧" },
  { code: "de", name: "Germany", nameHe: "גרמניה", flag: "🇩🇪" },
  { code: "fr", name: "France", nameHe: "צרפת", flag: "🇫🇷" },
];

export const LIVE_MEDIA_LANGUAGES: { code: string; name: string; nameHe: string }[] = [
  { code: "heb", name: "Hebrew", nameHe: "עברית" },
  { code: "eng", name: "English", nameHe: "אנגלית" },
  { code: "ara", name: "Arabic", nameHe: "ערבית" },
  { code: "rus", name: "Russian", nameHe: "רוסית" },
  { code: "fra", name: "French", nameHe: "צרפתית" },
  { code: "deu", name: "German", nameHe: "גרמנית" },
];
