import {
  USER_PRESENTATION_QUERIES,
  type UserPresentationQuery,
  type UserPresentationQueryGroup,
} from "./userPresentationQueries";

export type QaQueryOverrides = {
  hiddenBuiltinIds: string[];
  customQueries: UserPresentationQuery[];
  editedBuiltin: Record<string, Partial<Pick<UserPresentationQuery, "category" | "prompt" | "group">>>;
};

export const QA_QUERIES_STORAGE_KEY = "grovee-presentation-qa-queries-v1";

const emptyOverrides = (): QaQueryOverrides => ({
  hiddenBuiltinIds: [],
  customQueries: [],
  editedBuiltin: {},
});

export function loadQaQueryOverrides(): QaQueryOverrides {
  try {
    const raw = localStorage.getItem(QA_QUERIES_STORAGE_KEY);
    if (!raw) return emptyOverrides();
    const parsed = JSON.parse(raw) as QaQueryOverrides;
    return {
      hiddenBuiltinIds: parsed.hiddenBuiltinIds ?? [],
      customQueries: parsed.customQueries ?? [],
      editedBuiltin: parsed.editedBuiltin ?? {},
    };
  } catch {
    return emptyOverrides();
  }
}

export function saveQaQueryOverrides(overrides: QaQueryOverrides): void {
  localStorage.setItem(QA_QUERIES_STORAGE_KEY, JSON.stringify(overrides));
}

export function loadEffectiveQueries(overrides = loadQaQueryOverrides()): UserPresentationQuery[] {
  const hidden = new Set(overrides.hiddenBuiltinIds);
  const builtin = USER_PRESENTATION_QUERIES.filter((q) => !hidden.has(q.id)).map((q) => {
    const edit = overrides.editedBuiltin[q.id];
    if (!edit) return q;
    return { ...q, ...edit };
  });
  const custom = overrides.customQueries.map((q) => ({ ...q, custom: true as const }));
  return [...builtin, ...custom];
}

export function hideBuiltinQuery(id: string, overrides = loadQaQueryOverrides()): QaQueryOverrides {
  const next = { ...overrides, hiddenBuiltinIds: [...new Set([...overrides.hiddenBuiltinIds, id])] };
  saveQaQueryOverrides(next);
  return next;
}

export function restoreBuiltinQuery(id: string, overrides = loadQaQueryOverrides()): QaQueryOverrides {
  const next = {
    ...overrides,
    hiddenBuiltinIds: overrides.hiddenBuiltinIds.filter((x) => x !== id),
    editedBuiltin: { ...overrides.editedBuiltin },
  };
  delete next.editedBuiltin[id];
  saveQaQueryOverrides(next);
  return next;
}

export function upsertCustomQuery(
  query: UserPresentationQuery,
  overrides = loadQaQueryOverrides(),
): QaQueryOverrides {
  const custom = overrides.customQueries.filter((q) => q.id !== query.id);
  const next = {
    ...overrides,
    customQueries: [...custom, { ...query, custom: true as const }],
  };
  saveQaQueryOverrides(next);
  return next;
}

export function deleteCustomQuery(id: string, overrides = loadQaQueryOverrides()): QaQueryOverrides {
  const next = {
    ...overrides,
    customQueries: overrides.customQueries.filter((q) => q.id !== id),
  };
  saveQaQueryOverrides(next);
  return next;
}

export function editQuery(
  id: string,
  patch: Partial<Pick<UserPresentationQuery, "category" | "prompt" | "group">>,
  overrides = loadQaQueryOverrides(),
): QaQueryOverrides {
  if (overrides.customQueries.some((q) => q.id === id)) {
    return upsertCustomQuery(
      { ...overrides.customQueries.find((q) => q.id === id)!, ...patch },
      overrides,
    );
  }
  const next = {
    ...overrides,
    editedBuiltin: {
      ...overrides.editedBuiltin,
      [id]: { ...overrides.editedBuiltin[id], ...patch },
    },
  };
  saveQaQueryOverrides(next);
  return next;
}

export function nextCustomQueryId(queries: UserPresentationQuery[]): string {
  let n = 1;
  while (queries.some((q) => q.id === `CUST-${String(n).padStart(3, "0")}`)) n++;
  return `CUST-${String(n).padStart(3, "0")}`;
}

export function resetQaQueryOverrides(): void {
  localStorage.removeItem(QA_QUERIES_STORAGE_KEY);
}

export function exportQaQueryOverrides(): string {
  return JSON.stringify(loadQaQueryOverrides(), null, 2);
}

export function importQaQueryOverrides(raw: string): QaQueryOverrides {
  const parsed = JSON.parse(raw) as QaQueryOverrides;
  const next: QaQueryOverrides = {
    hiddenBuiltinIds: parsed.hiddenBuiltinIds ?? [],
    customQueries: parsed.customQueries ?? [],
    editedBuiltin: parsed.editedBuiltin ?? {},
  };
  saveQaQueryOverrides(next);
  return next;
}

export const QA_GROUP_LABELS: Record<UserPresentationQueryGroup, string> = {
  basic: "יכולות בסיס",
  cross: "הצלבת מקורות",
  natural: "שאלות טבעיות",
  events: "אירועים / חריגות",
  ui: "בדיקת UI",
};
