import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };

type State = { error: Error | null };

/**
 * Catches React render errors (missing chunk after deploy, bad props, etc.)
 * so users see a message instead of a blank screen.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[GROVEE] React render error:", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      const msg = this.state.error.message;
      return (
        <div className="fatal-boot-error" role="alert">
          <h1>שגיאת ממשק</h1>
          <p>
            חבילת JS לא נטענה כמו שצריך (404, רשת, או גרסה ישנה ב-cache). פתח Console (F12) לפרטים.
          </p>
          <pre className="fatal-boot-pre">{msg}</pre>
          <button type="button" className="pill-button" onClick={() => window.location.reload()}>
            רענון הדף
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
