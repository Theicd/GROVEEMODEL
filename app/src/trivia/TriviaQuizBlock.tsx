import type { TriviaOptionId, TriviaQuiz } from "./triviaParse";

type Props = {
  quiz: TriviaQuiz;
  compact?: boolean;
  selections?: Record<number, TriviaOptionId>;
  onPick: (questionNumber: number, optionId: TriviaOptionId) => void;
};

const OPT_LABEL_HE: Record<TriviaOptionId, string> = {
  A: "א",
  B: "ב",
  C: "ג",
  D: "ד",
};

export function TriviaQuizBlock({ quiz, compact, selections, onPick }: Props) {
  return (
    <div
      className={`trivia-quiz${compact ? " trivia-quiz--compact" : ""}`}
      dir="rtl"
      role="group"
      aria-label="חידון טריוויה"
    >
      <header className="trivia-quiz-header">
        <span className="trivia-quiz-badge">🎯 טריוויה</span>
        {quiz.title ? <span className="trivia-quiz-title">{quiz.title}</span> : null}
        <span className="trivia-quiz-count">{quiz.questions.length} שאלות</span>
      </header>

      <ol className="trivia-quiz-list">
        {quiz.questions.map((q) => {
          const picked = selections?.[q.number];
          return (
            <li key={q.number} className="trivia-quiz-card">
              <div className="trivia-quiz-card-head">
                <span className="trivia-quiz-qnum">שאלה {q.number}</span>
              </div>
              <p className="trivia-quiz-text">{q.text}</p>
              <div className="trivia-quiz-options" role="listbox" aria-label={`אפשרויות לשאלה ${q.number}`}>
                {q.options.map((opt) => {
                  const heLabel = OPT_LABEL_HE[opt.id];
                  const active = picked === opt.id;
                  return (
                    <button
                      key={`${q.number}-${opt.id}`}
                      type="button"
                      role="option"
                      aria-selected={active}
                      className={`trivia-quiz-option${active ? " is-picked" : ""}`}
                      onClick={() => onPick(q.number, opt.id)}
                    >
                      <span className="trivia-quiz-opt-id" aria-hidden="true">
                        {heLabel}
                      </span>
                      <span className="trivia-quiz-opt-text">{opt.text}</span>
                    </button>
                  );
                })}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
