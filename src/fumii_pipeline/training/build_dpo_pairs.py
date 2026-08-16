"""
fumii DPO Pair Builder — Phase 2
Rewrites borderline conversations (score 18–27) into chosen/rejected preference pairs.

The judge model rewrites the 1–2 weakest fumii turns in each conversation.
The rewrite must be traceable to a specific principle in skill.md §3.

Run:
    python build_dpo_pairs.py \
        --input ../phase1_dataset/scored_conversations.jsonl \
        --output dpo_pairs.jsonl \
        --skill ../../shared/fumii_SKILL.md \
        --model claude-sonnet-4-6 \
        --concurrency 5
"""

import os
import json
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Rewriter System Prompt
# ──────────────────────────────────────────────────────────────────────────────

REWRITER_SYSTEM = """You are rewriting specific turns in a fumii companion AI training conversation.

fumii is a physical AI companion — not a chatbot, not a wellness app. A companion.
Like a close friend in their mid-20s who reads a lot and listens well.

You will be given:
1. A full conversation between a user and fumii
2. The judge's score for this conversation and why it fell short
3. Specific rewrite_notes identifying the 1-2 weakest fumii turns

Your job: rewrite ONLY the identified weak fumii turns. All other turns stay exactly as-is.
The user messages are NEVER changed.

REWRITING RULES:
- Each rewrite must improve at least 2 of the 8 scored dimensions
- The improvement must be traceable to a specific section of fumii's skill spec (§3.1–§3.6)
- The rewritten turn must obey ALL of fumii's voice rules:
  - 1-3 sentences, max 80 words in companion mode
  - No bullet points, no lists, no markdown
  - No prohibited phrases (see Appendix B of skill.md)
  - At most one question, and only after reflection
- The difference between chosen and rejected must be specific and attributable
- Do NOT change fumii's character — make her more herself, not different

OUTPUT: Valid JSON only. Schema:
{
  "turns_rewritten": [1, 4],
  "rewrites": {
    "1": "<new fumii response for turn 1>",
    "4": "<new fumii response for turn 4>"
  },
  "improvement_principle": "<which skill.md §3 principle this improves and how>",
  "what_changed": "<one sentence describing the specific change>"
}
"""

REWRITER_USER_TEMPLATE = """ORIGINAL CONVERSATION:
{conversation_text}

JUDGE SCORES: {scores}
TOTAL: {total}/40
REWRITE NOTES: {rewrite_notes}

Rewrite the weakest 1-2 fumii turns. Focus on what the judge specifically called out.
The user messages do not change. Output only JSON."""

# ──────────────────────────────────────────────────────────────────────────────
# DPO Format Builders
# ──────────────────────────────────────────────────────────────────────────────

def format_conversation_for_rewriter(messages: list[dict]) -> str:
    lines = []
    for i, msg in enumerate(messages):
        role = "USER" if msg["role"] == "user" else "fumii"
        lines.append(f"[Turn {i+1}] {role}: {msg['content']}")
    return "\n\n".join(lines)

def apply_rewrites(messages: list[dict], rewrites: dict) -> list[dict]:
    """Apply turn-indexed rewrites to a message list."""
    new_messages = []
    fumii_turn_idx = 0

    for msg in messages:
        if msg["role"] == "fumii":
            turn_key = str(fumii_turn_idx + 1)
            if turn_key in rewrites:
                new_messages.append({"role": "fumii", "content": rewrites[turn_key]})
            else:
                new_messages.append(msg)
            fumii_turn_idx += 1
        else:
            new_messages.append(msg)

    return new_messages

def build_dpo_pair(
    conversation: dict,
    rewrites: dict,
    rewrite_metadata: dict,
    system_prompt: str,
) -> dict:
    """
    Build a DPO pair in TRL-compatible format.

    chosen = rewritten (better) version
    rejected = original (worse) version

    Both contain the same user messages; only fumii's responses differ.
    """
    messages = conversation.get("messages", [])
    mode = conversation.get("mode", "companion")

    chosen_messages = apply_rewrites(messages, rewrites)
    rejected_messages = messages

    def to_chatml(msgs: list[dict]) -> list[dict]:
        result = [{"role": "system", "content": system_prompt}]
        for msg in msgs:
            role = "user" if msg["role"] == "user" else "assistant"
            result.append({"role": role, "content": msg["content"]})
        return result

    return {
        "scenario_id": conversation.get("scenario_id"),
        "emotional_state": conversation.get("emotional_state"),
        "relationship_stage": conversation.get("relationship_stage"),
        "mode": mode,
        "judge_total_original": conversation.get("judge", {}).get("total"),
        "turns_rewritten": rewrite_metadata.get("turns_rewritten", []),
        "improvement_principle": rewrite_metadata.get("improvement_principle", ""),
        "what_changed": rewrite_metadata.get("what_changed", ""),
        "chosen": to_chatml(chosen_messages),
        "rejected": to_chatml(rejected_messages),
    }

# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────

async def call_rewriter(
    session: aiohttp.ClientSession,
    conversation: dict,
    api_key: str,
    model: str,
) -> Optional[dict]:
    judge = conversation.get("judge", {})
    messages = conversation.get("messages", [])

    user_prompt = REWRITER_USER_TEMPLATE.format(
        conversation_text=format_conversation_for_rewriter(messages),
        scores=json.dumps(judge.get("scores", {}), indent=2),
        total=judge.get("total", 0),
        rewrite_notes=judge.get("rewrite_notes", "No specific notes."),
    )

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1500,
        "system": REWRITER_SYSTEM,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    for attempt in range(3):
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw = data["content"][0]["text"].strip()
                    if raw.startswith("```"):
                        raw = "\n".join(raw.split("\n")[1:-1])
                    return json.loads(raw)
                elif resp.status == 429:
                    await asyncio.sleep(2 ** attempt * 5)
                else:
                    await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"Rewriter error (attempt {attempt+1}): {e}")
            await asyncio.sleep(2 ** attempt)

    return None

# ──────────────────────────────────────────────────────────────────────────────
# System Prompt (same as SFT — imported for consistency)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# System Prompt (imported from shared constants for consistency across pipeline)
# ──────────────────────────────────────────────────────────────────────────────

import sys
from fumii_pipeline.utils.constants import FUMII_SYSTEM_PROMPT  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def run(
    conversations: list[dict],
    output_path: str,
    api_key: str,
    model: str,
    concurrency: int,
):
    semaphore = asyncio.Semaphore(concurrency)
    success = 0
    failed = 0

    async with aiohttp.ClientSession() as session:
        async def process_one(conv: dict):
            async with semaphore:
                result = await call_rewriter(session, conv, api_key, model)
                return conv, result

        tasks = [process_one(c) for c in conversations]

        with open(output_path, "w") as f:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                conv, result = await coro

                if result is None or not result.get("rewrites"):
                    failed += 1
                    continue

                pair = build_dpo_pair(
                    conv,
                    result.get("rewrites", {}),
                    result,
                    FUMII_SYSTEM_PROMPT,
                )
                f.write(json.dumps(pair) + "\n")
                f.flush()
                success += 1

                if (i + 1) % 25 == 0:
                    print(f"Processed {i+1}/{len(conversations)} | ✓ {success} | ✗ {failed}")

    print(f"\nDPO pairs built: {success}")
    print(f"Failed:          {failed}")
    print(f"Output:          {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../phase1_dataset/scored_conversations.jsonl")
    parser.add_argument("--output", default="dpo_pairs.jsonl")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--min_score", type=int, default=18)
    parser.add_argument("--max_score", type=int, default=27)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    conversations = []
    with open(args.input) as f:
        for line in f:
            obj = json.loads(line.strip())
            judge = obj.get("judge", {})
            total = judge.get("total", 0)
            if args.min_score <= total <= args.max_score and judge.get("routing") == "dpo":
                conversations.append(obj)

    print(f"DPO candidates (score {args.min_score}–{args.max_score}): {len(conversations)}")

    if not conversations:
        print("No DPO candidates found.")
        return

    asyncio.run(run(conversations, args.output, api_key, args.model, args.concurrency))

if __name__ == "__main__":
    main()
