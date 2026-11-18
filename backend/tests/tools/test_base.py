import numpy as np
import pandas as pd
from pydantic import BaseModel

from adsci.tools.base import Tool, ToolOutput, ToolRegistry, summarize_frame


class EchoInput(BaseModel):
    text: str
    times: int


class Echo(Tool):
    name = "echo"
    description = "Echo text."
    Input = EchoInput

    async def run(self, args, ctx):
        return ToolOutput(summary=args.text * args.times)


class Boom(Tool):
    name = "boom"
    description = "Always fails."
    Input = EchoInput

    async def run(self, args, ctx):
        raise RuntimeError("kaboom")


async def test_registry_validation_error_names_fields(ctx):
    reg = ToolRegistry([Echo()])
    out = await reg.execute("echo", {"text": "a"}, ctx)
    assert out.is_error and "times" in out.summary


async def test_registry_unknown_tool(ctx):
    out = await ToolRegistry([Echo()]).execute("nope", {}, ctx)
    assert out.is_error and "echo" in out.summary


async def test_registry_exception_becomes_error_output(ctx):
    out = await ToolRegistry([Boom()]).execute("boom", {"text": "a", "times": 1}, ctx)
    assert out.is_error and "kaboom" in out.summary


async def test_registry_success_and_specs(ctx):
    reg = ToolRegistry([Echo()])
    assert (await reg.execute("echo", {"text": "ab", "times": 2}, ctx)).summary == "abab"
    [spec] = reg.specs()
    assert spec.name == "echo" and spec.parameters["required"] == ["text", "times"]


def test_summarize_huge_frame_stays_small():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(100_000, 300)), columns=[f"c{i}" for i in range(300)])
    text = summarize_frame(df)
    assert len(text) < 4000 and "100000 rows" in text and "300 columns" in text


def test_summarize_small_frame_lists_rows():
    text = summarize_frame(pd.DataFrame({"region": ["West", "East"], "aov": [80.5, 70.25]}))
    assert "West" in text and "80.5" in text
