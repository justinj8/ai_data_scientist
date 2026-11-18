import type {
  ConnectionConfig,
  Insight,
  LoadReport,
  ModelChoice,
  ModelInfo,
  Overview,
  ProviderStatus,
  QuestionRecord,
  SourceInfo,
  TableInfo,
  UIBlock,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let message = `Request failed (${res.status}).`;
    try {
      const detail = (await res.json()).detail;
      if (typeof detail === "string") message = detail;
      else if (Array.isArray(detail)) message = detail.map((d) => d.msg).join("; ");
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("json") ? res.json() : res.text()) as Promise<T>;
}

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body) });

export const api = {
  createSession: () => request<{ session_id: string }>("/api/sessions", { method: "POST" }),
  schema: (sid: string) => request<{ sources: SourceInfo[] }>(`/api/sessions/${sid}/schema`),
  upload: (sid: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ tables: LoadReport[] }>(`/api/sessions/${sid}/files`, { method: "POST", body: form });
  },
  loadSample: (sid: string) => request<{ sources: string[] }>(`/api/sessions/${sid}/samples/retail`, { method: "POST" }),
  testConnection: (config: ConnectionConfig) =>
    request<{ ok: boolean; error: string | null }>("/api/connections/test", { method: "POST", ...json(config) }),
  addConnection: (sid: string, name: string, config: ConnectionConfig) =>
    request<{ name: string; tables: TableInfo[] }>(`/api/sessions/${sid}/connections`, {
      method: "POST",
      ...json({ name, config }),
    }),
  removeSource: (sid: string, name: string) =>
    request<void>(`/api/sessions/${sid}/sources/${encodeURIComponent(name)}`, { method: "DELETE" }),
  overview: (sid: string, source: string, table: string) =>
    request<Overview>(
      `/api/sessions/${sid}/overview?source=${encodeURIComponent(source)}&table=${encodeURIComponent(table)}`,
    ),
  questions: (sid: string) => request<{ questions: QuestionRecord[] }>(`/api/sessions/${sid}/questions`),
  insights: (sid: string) => request<{ insights: Insight[] }>(`/api/sessions/${sid}/insights`),
  pin: (sid: string, title: string, block: UIBlock, questionId?: string | null) =>
    request<Insight>(`/api/sessions/${sid}/insights`, {
      method: "POST",
      ...json({ title, block, question_id: questionId ?? null }),
    }),
  updateInsight: (sid: string, id: string, patch: { title?: string; note?: string; order?: number }) =>
    request<Insight>(`/api/sessions/${sid}/insights/${id}`, { method: "PATCH", ...json(patch) }),
  unpin: (sid: string, id: string) => request<void>(`/api/sessions/${sid}/insights/${id}`, { method: "DELETE" }),
  models: () => request<{ default: string; choices: ModelChoice[] }>("/api/models"),
  allModels: () => request<{ models: ModelInfo[] }>("/api/models/all"),
  credentials: () => request<{ providers: Record<string, ProviderStatus> }>("/api/credentials"),
  setApiKey: (provider: string, apiKey: string) =>
    request<{ ok: boolean }>(`/api/credentials/${provider}/api-key`, { method: "PUT", ...json({ api_key: apiKey }) }),
  removeCredentials: (provider: string, kind?: string) =>
    request<void>(`/api/credentials/${provider}${kind ? `?kind=${kind}` : ""}`, { method: "DELETE" }),
  chatgptStart: () => request<{ authorize_url: string }>("/api/auth/chatgpt/start", { method: "POST" }),
  grokStart: () =>
    request<{ verification_uri: string; verification_uri_complete?: string; user_code: string; interval: number }>(
      "/api/auth/grok/start",
      { method: "POST" },
    ),
  grokStatus: () => request<{ status: "pending" | "approved" | "denied" | "expired" }>("/api/auth/grok/status"),
  logout: (provider: string) => request<{ ok: boolean }>(`/api/auth/${provider}/logout`, { method: "POST" }),
  notebookUrl: (sid: string) => `/api/sessions/${sid}/export/notebook`,
  reportUrl: (sid: string) => `/api/sessions/${sid}/export/report`,
};
