"""Build provider SSE bodies for adapter tests."""

import json


def sse(events: list[tuple[str | None, dict]]) -> bytes:
    out = []
    for name, data in events:
        if name:
            out.append(f"event: {name}")
        out.append(f"data: {json.dumps(data)}")
        out.append("")
    return ("\n".join(out) + "\n").encode()


def anthropic_tool_turn() -> bytes:
    msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
           "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 1}}
    return sse([
        ("message_start", {"type": "message_start", "message": msg}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "signature_delta", "signature": "sig123"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": "Plan: 1. "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": "query"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("content_block_start", {"type": "content_block_start", "index": 2,
                                 "content_block": {"type": "tool_use", "id": "toolu_1", "name": "run_sql", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 2,
                                 "delta": {"type": "input_json_delta", "partial_json": "{\"source\": \"files\", "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 2,
                                 "delta": {"type": "input_json_delta", "partial_json": "\"query\": \"select 1\"}"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                           "usage": {"output_tokens": 500}}),
        ("message_stop", {"type": "message_stop"}),
    ])


def anthropic_refusal_turn() -> bytes:
    msg = {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
           "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}
    return sse([
        ("message_start", {"type": "message_start", "message": msg}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_sequence": None},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ])
