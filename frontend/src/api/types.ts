// Mirrors the backend's pydantic models (adsci.*). Keep in sync with the API.

export type Metric = { label: string; value: string; detail?: string | null };
export type TableBlock = { type: "table"; columns: string[]; rows: unknown[][]; total_rows: number };
export type TextBlock = { type: "text"; text: string; format: "plain" | "markdown" };
export type ChartBlock = { type: "chart"; figure: PlotlyFigure; title?: string | null };
export type ImageBlock = { type: "image"; png_b64: string };
export type MetricsBlock = { type: "metrics"; items: Metric[] };
export type ErrorBlock = { type: "error"; message: string; detail?: string | null };
export type UIBlock = TableBlock | TextBlock | ChartBlock | ImageBlock | MetricsBlock | ErrorBlock;
export type PlotlyFigure = { data: Record<string, unknown>[]; layout?: Record<string, unknown> };

export type ColumnInfo = { name: string; dtype: string; nullable: boolean; original_name?: string | null };
export type TableInfo = { name: string; columns: ColumnInfo[]; row_count?: number | null; original_name?: string | null };
export type SourceInfo = { name: string; kind: "file" | "database"; dialect: string; tables: TableInfo[] };
export type LoadReport = { table: string; rows: number; warnings: string[] };

export type Engine = "postgresql" | "mysql" | "sqlite" | "mssql" | "snowflake" | "bigquery";
export type ConnectionConfig = {
  engine: Engine;
  host?: string;
  port?: number;
  database?: string;
  username?: string;
  password?: string;
  ssl?: boolean;
  account?: string;
  warehouse?: string;
  role?: string;
  db_schema?: string;
  project?: string;
  dataset?: string;
  credentials_json?: string;
  url?: string;
};

export type Provider = "anthropic" | "openai" | "xai";
export type ModelInfo = {
  id: string;
  provider: Provider | "fake";
  family: string;
  display_name: string;
  created?: string | null;
  supports_tools: boolean;
};
export type ChoiceKey = "opus" | "sonnet" | "gpt" | "grok";
export type ModelChoice = {
  key: ChoiceKey;
  label: string;
  model: ModelInfo;
  available: boolean;
  reason?: string | null;
  billed_to?: string | null;
};
export type ProviderStatus = {
  configured: boolean;
  kind?: string | null;
  masked?: string | null;
  account_label?: string | null;
  has_api_key: boolean;
  has_subscription: boolean;
  from_env: boolean;
  subscription_supported: boolean;
  subscription_experimental: boolean;
  subscription_available: boolean;
  subscription_note: string;
};

export type Usage = { input_tokens: number; output_tokens: number; cost_usd?: number | null; billed_to: string };
export type UserAction = "update_credentials" | "switch_model" | "retry" | "none";

export type AgentEvent =
  | { type: "model"; question_id: string; model: ModelInfo; billed_to?: string | null }
  | { type: "text_delta"; text: string }
  | { type: "plan"; text: string }
  | { type: "note"; text: string }
  | {
      type: "step_start";
      step_id: string;
      tool: string;
      purpose?: string | null;
      code?: string | null;
      language?: "sql" | "python" | null;
      args: Record<string, unknown>;
    }
  | {
      type: "step_output";
      step_id: string;
      blocks: UIBlock[];
      is_error: boolean;
      duration_s: number;
      code?: string | null;
      language?: "sql" | "python" | null;
    }
  | { type: "answer"; text: string; revised: boolean }
  | { type: "verification"; status: "verified" | "warnings"; ungrounded: string[]; issues: string[]; notes: string[] }
  | { type: "usage"; usage: Usage; model_id: string }
  | { type: "done"; question_id: string; answer: string; steps: number }
  | { type: "error"; message: string; user_action: UserAction };

export type Insight = {
  id: string;
  question_id?: string | null;
  title: string;
  block: UIBlock;
  note?: string | null;
  order: number;
  created_at: string;
};

export type DataQualityWarning = { column: string | null; kind: string; message: string };
export type Overview = {
  metrics: Metric[];
  warnings: DataQualityWarning[];
  distributions: ChartBlock[];
  correlation: ChartBlock | null;
  columns: { name: string; semantic: string; missing_pct: number; n_unique: number }[];
  sampled: boolean;
};

export type StepRecord = {
  step_id: string;
  tool: string;
  purpose?: string | null;
  code?: string | null;
  language?: string | null;
  blocks: UIBlock[];
  is_error: boolean;
};
export type QuestionRecord = {
  id: string;
  question: string;
  model_id: string;
  plan?: string | null;
  steps: StepRecord[];
  answer: string;
  verification?: { status: "verified" | "warnings"; ungrounded: string[]; issues: string[]; notes: string[] } | null;
};
