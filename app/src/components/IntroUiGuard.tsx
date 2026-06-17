import { useEffect, useState } from "react";

const NEW_DEV_URL = "http://127.0.0.1:5180/";
const NEW_PAGES_HINT = "npm run build:pages-docs && git push (docs/)";

/** Warn / redirect when the old clone or stale GitHub Pages bundle is loaded. */
export function IntroUiGuard() {
  const [issue, setIssue] = useState<"none" | "main-clone" | "stale-pages" | "wrong-port">("none");

  useEffect(() => {
    const { hostname, port, pathname } = window.location;
    const onGithubPages = /(?:^|\.)github\.io$/i.test(hostname) && pathname.includes("/GROVEEMODEL");

    if (onGithubPages) {
      const t = window.setTimeout(() => {
        if (!document.querySelector('[data-ui="hal-space-v2"]')) setIssue("stale-pages");
      }, 1200);
      return () => window.clearTimeout(t);
    }

    if (hostname === "127.0.0.1" && port === "5174") {
      window.location.replace(NEW_DEV_URL);
      setIssue("wrong-port");
      return;
    }

    const t = window.setTimeout(() => {
      const body = document.body.innerText;
      if (body.includes("טען מודל מקומי")) {
        setIssue("main-clone");
        return;
      }
      if ((port === "5173" || port === "5174") && !document.querySelector('[data-ui="hal-space-v2"]')) {
        setIssue("wrong-port");
      }
    }, 800);

    return () => window.clearTimeout(t);
  }, []);

  if (issue === "none") return null;

  const message =
    issue === "stale-pages"
      ? "GitHub Pages עדיין מציג build ישן. צריך push של docs/ המעודכן."
      : issue === "main-clone"
        ? "זה GROVEEMODEL-main (עיצוב ישן) — לא התיקייה שעדכנו."
        : "נפתח פורט/שרת dev לא נכון.";

  return (
    <div className="intro-ui-guard" role="alert">
      <div className="intro-ui-guard__panel">
        <p className="intro-ui-guard__title">עיצוב ישן — לא GROVEEMODEL החדש</p>
        <p className="intro-ui-guard__msg">{message}</p>
        {issue !== "stale-pages" ? (
          <a className="intro-ui-guard__link" href={NEW_DEV_URL}>
            פתח את הגרסה החדשה: {NEW_DEV_URL}
          </a>
        ) : (
          <p className="intro-ui-guard__hint" dir="ltr">
            {NEW_PAGES_HINT}
          </p>
        )}
        <p className="intro-ui-guard__signs">
          חדש = תג <strong>HAL·SPACE</strong> + כפתור «טען מודל לדפדפן»
        </p>
      </div>
    </div>
  );
}
