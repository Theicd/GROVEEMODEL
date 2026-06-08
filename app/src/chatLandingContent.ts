export type LandingSuggestion = {
  icon: string;
  label: string;
  prompt: string;
};

export const LANDING_HEADLINES = [
  "What would you like to create?",
  "What are we building today?",
  "Let's turn ideas into reality!",
  "What would you like to explore?",
  "What challenge can I solve for you today?",
  "Let's create something amazing.",
  "What's the plan today?",
] as const;

export const LANDING_SUGGESTION_SETS: LandingSuggestion[][] = [
  [
    { icon: "💻", label: "Create a web app", prompt: "Create a web application" },
    { icon: "📷", label: "Analyze an image", prompt: "Analyze an image" },
    {
      icon: "😊",
      label: "Detect emotions",
      prompt: "Detect emotions and expressions from the camera",
    },
  ],
  [
    { icon: "🎮", label: "Build a browser game", prompt: "Build a browser game" },
    {
      icon: "📷",
      label: "Describe the camera view",
      prompt: "Describe what the camera sees right now",
    },
    { icon: "✋", label: "Recognize hand gestures", prompt: "Recognize hand gestures from the camera" },
  ],
  [
    {
      icon: "💻",
      label: "HTML, CSS & JavaScript",
      prompt: "Generate HTML, CSS and JavaScript for a modern landing page",
    },
    {
      icon: "📷",
      label: "Analyze photos",
      prompt: "Analyze this photo and describe what you see in detail",
    },
    {
      icon: "🎤",
      label: "Voice + camera chat",
      prompt: "Let's talk — use the camera and respond naturally",
    },
  ],
  [
    { icon: "💻", label: "Create a web app", prompt: "Create a web application" },
    { icon: "📷", label: "Analyze images", prompt: "Analyze images and photos I attach" },
    {
      icon: "✋",
      label: "Gestures & expressions",
      prompt: "Recognize gestures and facial expressions from the camera",
    },
  ],
];
