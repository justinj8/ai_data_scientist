import type { SourceInfo } from "../../api/types";

export const TOOL_LABELS: Record<string, string> = {
  run_sql: "SQL query",
  run_python: "Python",
  create_chart: "Chart",
  profile_dataset: "Data profile",
  statistical_test: "Statistical test",
  train_model: "Predictive model",
  forecast: "Forecast",
  detect_anomalies: "Anomaly check",
  segment: "Segmentation",
};

/** Suggested first questions, based on what's loaded. */
export function starterQuestions(sources: SourceInfo[]): string[] {
  const tables = sources.flatMap((s) => s.tables.map((t) => ({ source: s.name, ...t })));
  const names = new Set(tables.map((t) => t.name));
  if (names.has("orders") && names.has("customers")) {
    return [
      "Which region has the highest average order value, and is the difference significant?",
      "What predicts customer churn, and how well can we predict it?",
      "Forecast daily revenue for the next 30 days.",
      "Were there any unusual days in daily revenue?",
    ];
  }
  const out: string[] = [];
  for (const t of tables.slice(0, 2)) {
    out.push(`Profile the ${t.name} table and flag any data-quality issues.`);
    const date = t.columns.find((c) => /date|time/i.test(c.dtype));
    const num = t.columns.find((c) => /int|float|double|decimal|numeric|real/i.test(c.dtype) && !/id$/i.test(c.name));
    if (date && num) out.push(`How has ${num.name} in ${t.name} changed over time?`);
  }
  return out.slice(0, 4);
}
