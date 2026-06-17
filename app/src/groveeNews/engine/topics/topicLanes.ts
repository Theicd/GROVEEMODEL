// @ts-nocheck
/** Single-keyword lanes — multi-word queries AND-match and return few hits. */
export type TopicLane = {
  id: string;
  label: string;
  query: string;
  icon: string;
};

export const TOPICS_PER_LANE = 2;

export const TOPIC_LANES: TopicLane[] = [
  // Geo & conflict
  { id: "israel", label: "Israel & Middle East", query: "israel", icon: "🇮🇱" },
  { id: "ukraine", label: "Ukraine", query: "ukraine", icon: "🇺🇦" },
  { id: "russia", label: "Russia", query: "russia", icon: "🇷🇺" },
  { id: "china", label: "China", query: "china", icon: "🇨🇳" },
  { id: "iran", label: "Iran", query: "iran", icon: "🇮🇷" },
  { id: "turkey", label: "Turkey", query: "turkey", icon: "🇹🇷" },
  { id: "africa", label: "Africa", query: "africa", icon: "🌍" },
  { id: "india", label: "India & South Asia", query: "india", icon: "🇮🇳" },
  { id: "latam", label: "Latin America", query: "brazil", icon: "🌎" },
  { id: "war", label: "War & Conflict", query: "war", icon: "⚔️" },
  { id: "diplomacy", label: "Diplomacy", query: "diplomacy", icon: "🕊️" },
  // Tech & science
  { id: "ai", label: "AI & ML", query: "ai", icon: "🤖" },
  { id: "tech", label: "Technology", query: "tech", icon: "💻" },
  { id: "cyber", label: "Cybersecurity", query: "cyber", icon: "🔐" },
  { id: "startup", label: "Startups", query: "startup", icon: "💡" },
  { id: "robotics", label: "Robotics", query: "robot", icon: "🦾" },
  { id: "biotech", label: "Biotech", query: "biotech", icon: "🧬" },
  { id: "space", label: "Space", query: "space", icon: "🚀" },
  { id: "science", label: "Science", query: "science", icon: "🔬" },
  { id: "nuclear", label: "Nuclear", query: "nuclear", icon: "☢️" },
  // Business & energy
  { id: "market", label: "Markets", query: "market", icon: "📈" },
  { id: "crypto", label: "Crypto", query: "crypto", icon: "₿" },
  { id: "energy", label: "Energy & Oil", query: "energy", icon: "⚡" },
  { id: "car", label: "Cars & EV", query: "car", icon: "🚗" },
  { id: "aviation", label: "Aviation", query: "aviation", icon: "✈️" },
  { id: "maritime", label: "Shipping & Trade", query: "shipping", icon: "🚢" },
  // Society & culture
  { id: "politics", label: "Politics", query: "politics", icon: "🏛️" },
  { id: "crime", label: "Crime & Justice", query: "crime", icon: "⚖️" },
  { id: "health", label: "Health", query: "health", icon: "🏥" },
  { id: "climate", label: "Climate", query: "climate", icon: "🌡️" },
  { id: "environment", label: "Environment", query: "environment", icon: "🌱" },
  { id: "education", label: "Education", query: "education", icon: "📚" },
  { id: "religion", label: "Religion", query: "religion", icon: "🕌" },
  { id: "travel", label: "Travel", query: "travel", icon: "🧳" },
  { id: "food", label: "Food & Dining", query: "food", icon: "🍽️" },
  { id: "fashion", label: "Fashion", query: "fashion", icon: "👗" },
  { id: "music", label: "Music", query: "music", icon: "🎵" },
  { id: "film", label: "Film & TV", query: "film", icon: "🎬" },
  { id: "gaming", label: "Gaming", query: "gaming", icon: "🎮" },
  { id: "sport", label: "Sports", query: "sport", icon: "⚽" },
  { id: "tcm", label: "TCM & Alternative", query: "tcm", icon: "🌿" },
];

export const TOPIC_LANE_BY_ID = new Map(TOPIC_LANES.map((l) => [l.id, l]));
