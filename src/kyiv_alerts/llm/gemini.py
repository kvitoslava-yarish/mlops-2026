import os
import time
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class Call:
    parsed: BaseModel | None
    text: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    tokens_thoughts: int
    model_version: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


def client():
    from google import genai

    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise SystemExit("GEMINI_API_KEY is empty: put the key in .env")
    return genai.Client(api_key=key)


def generate(cli, llm_cfg: dict, prompt: str, schema: type[BaseModel], model: str = "") -> Call:
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=llm_cfg["temperature"],
        max_output_tokens=llm_cfg["max_output_tokens"],
        response_mime_type="application/json",
        response_schema=schema,
    )
    if llm_cfg.get("thinking_budget") is not None:
        config.thinking_config = types.ThinkingConfig(thinking_budget=llm_cfg["thinking_budget"])
    start = time.perf_counter()
    try:
        resp = cli.models.generate_content(
            model=model or llm_cfg["model"], contents=prompt, config=config
        )
    except Exception as e:  # noqa: BLE001 — a failed call is recorded, not fatal for the run
        return Call(None, "", time.perf_counter() - start, 0, 0, 0, error=str(e)[:500])
    latency = time.perf_counter() - start
    usage = resp.usage_metadata
    return Call(
        parsed=resp.parsed,
        text=resp.text or "",
        latency_s=latency,
        tokens_in=usage.prompt_token_count or 0,
        tokens_out=usage.candidates_token_count or 0,
        tokens_thoughts=usage.thoughts_token_count or 0,
        model_version=getattr(resp, "model_version", "") or "",
    )
