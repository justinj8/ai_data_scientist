import type { AgentEvent } from "./types";

/** Incremental Server-Sent Events parser: feed it text chunks, it calls back once per complete frame. */
export class SSEParser {
  private buffer = "";
  private readonly onEvent: (e: AgentEvent) => void;

  constructor(onEvent: (e: AgentEvent) => void) {
    this.onEvent = onEvent;
  }

  push(chunk: string): void {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    let idx: number;
    while ((idx = this.buffer.indexOf("\n\n")) !== -1) {
      const frame = this.buffer.slice(0, idx);
      this.buffer = this.buffer.slice(idx + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) this.onEvent(JSON.parse(data) as AgentEvent);
    }
  }
}

/** POST a question and stream the agent's events until the response ends. */
export async function streamAsk(
  sessionId: string,
  body: { question: string; model?: string | null },
  onEvent: (e: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api/sessions/${sessionId}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let message = `Request failed (${res.status}).`;
    try {
      const detail = (await res.json()).detail;
      if (detail) message = typeof detail === "string" ? detail : JSON.stringify(detail);
    } catch {
      /* keep the generic message */
    }
    onEvent({ type: "error", message, user_action: "retry" });
    return;
  }
  const parser = new SSEParser(onEvent);
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.push(value);
  }
}
