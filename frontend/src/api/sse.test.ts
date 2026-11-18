import { describe, expect, it } from "vitest";
import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("handles a frame split across chunks", () => {
    const got: unknown[] = [];
    const p = new SSEParser((e) => got.push(e));
    p.push('event: plan\ndata: {"type":"plan",');
    expect(got).toEqual([]);
    p.push('"text":"Plan: 1."}\n\n');
    expect(got).toEqual([{ type: "plan", text: "Plan: 1." }]);
  });

  it("handles two frames in one chunk, CRLF line endings and comments", () => {
    const got: unknown[] = [];
    const p = new SSEParser((e) => got.push(e));
    p.push(': ping\r\n\r\nevent: a\r\ndata: {"type":"text_delta","text":"x"}\r\n\r\ndata: {"type":"done","answer":"y"}\n\n');
    expect(got).toEqual([{ type: "text_delta", text: "x" }, { type: "done", answer: "y" }]);
  });

  it("joins multi-line data fields", () => {
    const got: unknown[] = [];
    const p = new SSEParser((e) => got.push(e));
    p.push('data: {"type":"note",\ndata: "text":"two lines"}\n\n');
    expect(got).toEqual([{ type: "note", text: "two lines" }]);
  });
});
