"""
fumii Shared Pipeline Utilities
Consolidates helpers that were duplicated across generate_dataset.py,
judge_model.py, and build_dpo_pairs.py.

Import from here instead of copy-pasting retry logic or JSON fence stripping.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Coroutine, List

# ──────────────────────────────────────────────────────────────────────────────
# JSONL I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def write_jsonl(records: List[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file (overwrites)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_jsonl(record: dict, path: str) -> None:
    """Append a single dict to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# JSON Fence Stripping
# Handles both ```json ... ``` and bare ``` ... ``` fences that LLMs emit.
# ──────────────────────────────────────────────────────────────────────────────

def strip_json_fences(text: str) -> str:
    """
    Strip markdown code fences from an LLM response before JSON parsing.

    Handles:
      ```json\\n{...}\\n```
      ```\\n{...}\\n```
      {bare json without fences}
    """
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.split("\n")
    # Remove opening fence line (``` or ```json)
    lines = lines[1:]
    # Remove closing fence line if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Async Retry
# ──────────────────────────────────────────────────────────────────────────────

async def async_retry(
    coro_factory: Callable[[], Coroutine],
    retries: int = 3,
    base_wait: float = 2.0,
    on_rate_limit_wait: float = 10.0,
) -> Any:
    """
    Retry an async coroutine factory with exponential backoff.

    Args:
        coro_factory: A callable that returns a new coroutine on each call.
                      (Pass a lambda or functools.partial — not a coroutine directly,
                      since coroutines can only be awaited once.)
        retries: Maximum number of attempts.
        base_wait: Base wait in seconds (doubles each attempt).
        on_rate_limit_wait: Extra wait on HTTP 429 responses.

    Returns:
        The return value of the coroutine on success.

    Raises:
        Exception: Re-raises the last exception after all retries are exhausted.

    Example:
        result = await async_retry(
            lambda: call_anthropic(session, system, user, model, api_key)
        )
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            wait = base_wait ** attempt
            await asyncio.sleep(wait)

    raise last_exc
