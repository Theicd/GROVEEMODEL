export type TriviaOptionId = "A" | "B" | "C" | "D";

export type TriviaOption = {
  id: TriviaOptionId;
  text: string;
};

export type TriviaQuestion = {
  number: number;
  text: string;
  options: TriviaOption[];
};

export type TriviaQuiz = {
  title?: string;
  questions: TriviaQuestion[];
};

const OPTION_LINE_RE = /^([A-Da-dא-ד])[\).:\-]\s*(.+)$/u;

const QUESTION_START_RE = /^(\d+)[\).:\-]\s*(.+)$|^שאלה\s*(\d+)\s*[:\-.]?\s*(.*)$/i;

const ANSWER_LEAK_RE =
  /^(?:the answer is|correct answer|התשובה(?:\s+היא|\s+הנכונה)|תשובה\s*:\s*|answers?\s*(?:are|:))/i;

const HEB_TO_LAT: Record<string, TriviaOptionId> = {
  א: "A",
  ב: "B",
  ג: "C",
  ד: "D",
};

function normalizeOptionId(raw: string): TriviaOptionId | null {
  const c = raw.trim().charAt(0);
  if (/^[A-Da-d]$/.test(c)) return c.toUpperCase() as TriviaOptionId;
  if (HEB_TO_LAT[c]) return HEB_TO_LAT[c];
  return null;
}

function isNoiseLine(line: string): boolean {
  const t = line.trim();
  if (!t) return true;
  if (ANSWER_LEAK_RE.test(t)) return true;
  if (/^Q\s*&\s*A|^question\s+\d+\s+to/i.test(t)) return true;
  if (/^[E-Fe-fא-ת][\).:\-]/u.test(t)) return true;
  return false;
}

/** Parse LLM trivia output into structured quiz (game-show card UI). */
export function parseTriviaQuiz(raw: string, maxQuestions = 5): TriviaQuiz | null {
  const lines = raw
    .replace(/\r/g, "")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !isNoiseLine(l));

  if (!lines.length) return null;

  const questions: TriviaQuestion[] = [];
  let current: TriviaQuestion | null = null;

  const pushCurrent = () => {
    if (!current) return;
    if (current.text.trim() && current.options.length >= 2) {
      questions.push({
        ...current,
        options: current.options.slice(0, 4),
      });
    }
    current = null;
  };

  for (const line of lines) {
    const qMatch = line.match(QUESTION_START_RE);
    if (qMatch) {
      pushCurrent();
      const num = Number.parseInt(qMatch[1] ?? qMatch[3] ?? "0", 10);
      const text = (qMatch[2] ?? qMatch[4] ?? "").trim();
      if (num > 0 && text) {
        current = { number: num, text, options: [] };
        continue;
      }
    }

    const oMatch = line.match(OPTION_LINE_RE);
    if (oMatch) {
      const idRaw = oMatch[1];
      const optText = (oMatch[2] ?? "").trim();
      const id = normalizeOptionId(idRaw);
      if (!id || !optText) continue;
      if (!current) {
        current = { number: questions.length + 1, text: "…", options: [] };
      }
      if (!current.options.some((o) => o.id === id)) {
        current.options.push({ id, text: optText });
      }
      continue;
    }

    if (current && current.options.length === 0 && !current.text.endsWith("…")) {
      current.text = `${current.text} ${line}`.trim();
    } else if (current && current.options.length === 0) {
      current.text = line;
    }
  }
  pushCurrent();

  const capped = questions.slice(0, maxQuestions);
  if (!capped.length) return null;
  return { questions: capped };
}

export function triviaQuizSummaryHe(quiz: TriviaQuiz): string {
  const topic = quiz.title?.trim();
  const n = quiz.questions.length;
  return topic
    ? `🎯 ${topic} — ${n} שאלות. בחר תשובה לכל שאלה:`
    : `🎯 חידון טריוויה — ${n} שאלות. בחר תשובה:`;
}
