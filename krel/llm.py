from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from . import config
from .io_utils import extract_json_text, parse_jsonish


class LLMConfigError(RuntimeError):
    pass


def make_client(api_key: str | None = None) -> OpenAI:
    api_key = api_key if api_key is not None else config.LLM_API_KEY
    if not api_key:
        raise LLMConfigError("Missing KREL_LLM_API_KEY. Set it in .env or the environment.")
    return OpenAI(api_key=api_key)


def chat_text(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 32768,
    temperature: float = 0.0,
    json_mode: bool = False,
    stream: bool = False,
    timeout: float | None = None,
) -> str:
    client = make_client(api_key=api_key)
    body: dict[str, Any] = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if timeout is not None:
        body["timeout"] = timeout
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if stream:
        body["stream"] = True
        answer_parts: list[str] = []
        completion = client.chat.completions.create(**body)
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                answer_parts.append(content)
        return "".join(answer_parts).strip()

    completion = client.chat.completions.create(**body)
    message = completion.choices[0].message
    content = getattr(message, "content", None) or ""
    return content.strip()


def chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 32768,
    temperature: float = 0.0,
    json_mode: bool = True,
    stream: bool = False,
    retries: int = 2,
    retry_sleep: float = 2.0,
) -> tuple[Any, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = chat_text(
                messages,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                stream=stream,
            )
            payload = parse_jsonish(raw)
            return payload, extract_json_text(raw)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"LLM JSON call failed: {last_error}") from last_error


def dumps_json_response(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
