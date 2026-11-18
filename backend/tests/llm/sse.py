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


def _response_obj(output, status="completed", incomplete=None, usage=(800, 300)):
    return {
        "id": "resp_1", "object": "response", "created_at": 1763000000, "model": "gpt-6.2-astra",
        "status": status, "incomplete_details": incomplete, "output": output, "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [], "error": None,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage),
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }


def openai_tool_turn() -> bytes:
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"}
    message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Plan: query", "annotations": []}]}
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "run_sql",
            "arguments": "{\"source\": \"files\", \"query\": \"select 1\"}", "status": "completed"}
    return sse([
        (None, {"type": "response.created", "sequence_number": 0, "response": _response_obj([], status="in_progress")}),
        (None, {"type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg_1", "output_index": 1,
                "content_index": 0, "delta": "Plan: ", "logprobs": []}),
        (None, {"type": "response.output_text.delta", "sequence_number": 2, "item_id": "msg_1", "output_index": 1,
                "content_index": 0, "delta": "query", "logprobs": []}),
        (None, {"type": "response.completed", "sequence_number": 3,
                "response": _response_obj([reasoning, message, call])}),
    ])


def openai_incomplete_turn() -> bytes:
    return sse([(None, {"type": "response.incomplete", "sequence_number": 0,
                        "response": _response_obj([], status="incomplete",
                                                  incomplete={"reason": "max_output_tokens"})})])
