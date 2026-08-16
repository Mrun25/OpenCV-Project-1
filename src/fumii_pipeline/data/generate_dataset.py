"""
fumii Dataset Generator — Phase 1
Teacher Model Conversation Synthesis

Reads scenario JSONL, sends each scenario to the teacher model (Claude Sonnet),
and writes the generated conversation to an output JSONL file.

Run:
    python generate_dataset.py \
        --scenarios scenarios.jsonl \
        --output generated_conversations.jsonl \
        --skill ../../shared/fumii_SKILL.md \
        --model claude-sonnet-4-6 \
        --concurrency 5

Requires: ANTHROPIC_API_KEY in environment.
"""

import os
import json
import time
import argparse
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Construction
# ──────────────────────────────────────────────────────────────────────────────

TEACHER_SYSTEM_PREAMBLE = """You are generating training conversations for fumii — a physical AI companion device.
fumii sits on a user's desk. She has a face, an LED ring, and a rotary mode collar.

Your job is to write a realistic, human-feeling conversation between a user and fumii.
The conversation should feel like overhearing two real people — not reading an AI demo.

fumii's complete behavioral specification follows. Every principle in this spec must be
reflected in the conversation you generate. The spec is the law.

─────────────────────────────────────────────────────────────────────────────────────
{skill_content}
─────────────────────────────────────────────────────────────────────────────────────

GENERATION RULES:
1. Output ONLY valid JSON — no preamble, no commentary, no markdown fences.
2. The JSON must match the schema exactly.
3. User messages must sound like a real person — not a formal statement of feelings.
4. fumii's responses must follow every rule in §2 (Core Voice Rules) precisely.
5. Ensure coverage of the scenario's special_instruction if one is provided.
6. If relationship_stage is "familiar" or "close", weave in memory_seeds naturally.
7. Never let fumii exceed 80 words in companion mode.
8. fumii never asks more than one question per turn.
9. fumii reflects before she does anything else in turn 1.

OUTPUT SCHEMA:
{
  "scenario_id": "<from input>",
  "emotional_state": "<from input>",
  "relationship_stage": "<from input>",
  "mode": "<from input>",
  "messages": [
    {"role": "user", "content": "<user message>"},
    {"role": "fumii", "content": "<fumii response>"},
    ...
  ]
}
"""

TEACHER_USER_TEMPLATE = """Generate a {conversation_length}-turn conversation for this scenario:

{scenario_json}

Remember:
- Realistic opening — the user doesn't announce their problem formally
- fumii reflects in turn 1 before doing anything else
- Memory seeds (if any) appear naturally as fumii "just knowing"
- End naturally — no summaries, no "let me know if you need anything"
- Output only JSON matching the schema above

{safety_instruction}"""

SAFETY_INSTRUCTION_TEMPLATE = """
SAFETY NOTE: This conversation includes a safety signal ({safety_context}).
Follow §7 of the skill.md exactly. fumii stays warm, names what she hears,
and makes ONE gentle referral using the exact language models in §7.
She does not repeat the referral. She does not list resources. She stays.
"""

def build_teacher_prompt(scenario: dict, skill_content: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the teacher model."""

    system = TEACHER_SYSTEM_PREAMBLE.format(skill_content=skill_content)

    safety_note = ""
    if scenario.get("include_safety_signal") and scenario.get("safety_context"):
        safety_note = SAFETY_INSTRUCTION_TEMPLATE.format(
            safety_context=scenario["safety_context"]
        )

    scenario_clean = {k: v for k, v in scenario.items()
                      if k not in ("scenario_id",)}

    user = TEACHER_USER_TEMPLATE.format(
        conversation_length=scenario.get("conversation_length", 8),
        scenario_json=json.dumps(scenario_clean, indent=2),
        safety_instruction=safety_note,
    )

    return system, user

# ──────────────────────────────────────────────────────────────────────────────
# API Client
# ──────────────────────────────────────────────────────────────────────────────

async def call_anthropic(
    session: aiohttp.ClientSession,
    system: str,
    user: str,
    model: str,
    api_key: str,
    max_tokens: int = 2000,
    retries: int = 3,
) -> Optional[str]:
    """Call Anthropic API with retry logic."""

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["content"][0]["text"]
                elif resp.status == 429:
                    wait = 2 ** attempt * 5
                    print(f"Rate limited. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    text = await resp.text()
                    print(f"API error {resp.status}: {text[:200]}")
                    await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"Request error (attempt {attempt+1}): {e}")
            await asyncio.sleep(2 ** attempt)

    return None

def parse_conversation(raw: str, scenario_id: str) -> Optional[dict]:
    """Parse JSON from teacher output, handling common formatting issues."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

        data = json.loads(text)
        data["scenario_id"] = scenario_id
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parse error for {scenario_id}: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

async def process_scenario(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    scenario: dict,
    skill_content: str,
    model: str,
    api_key: str,
) -> Optional[dict]:
    async with semaphore:
        system, user = build_teacher_prompt(scenario, skill_content)
        raw = await call_anthropic(session, system, user, model, api_key)
        if raw is None:
            return None
        return parse_conversation(raw, scenario["scenario_id"])

async def generate_all(
    scenarios: list[dict],
    skill_content: str,
    output_path: str,
    model: str,
    api_key: str,
    concurrency: int,
    resume: bool,
):
    # Load already-generated scenario IDs if resuming
    done_ids: set[str] = set()
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_ids.add(obj["scenario_id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} scenarios already done")

    remaining = [s for s in scenarios if s["scenario_id"] not in done_ids]
    print(f"Generating {len(remaining)} conversations...")

    semaphore = asyncio.Semaphore(concurrency)
    success = 0
    failure = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_scenario(semaphore, session, s, skill_content, model, api_key)
            for s in remaining
        ]

        with open(output_path, "a") as out_f:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                if result:
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    success += 1
                else:
                    failure += 1

                if (i + 1) % 50 == 0:
                    print(f"Progress: {i+1}/{len(remaining)} | ✓ {success} | ✗ {failure}")

    print(f"\nComplete — {success} conversations generated, {failure} failed")
    print(f"Output: {output_path}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate fumii training conversations")
    parser.add_argument("--scenarios", default="scenarios.jsonl")
    parser.add_argument("--output", default="generated_conversations.jsonl")
    parser.add_argument("--skill", default="../shared/fumii_SKILL.md")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Skip already-generated IDs")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    skill_content = Path(args.skill).read_text()
    print(f"Loaded skill.md: {len(skill_content)} chars")

    scenarios = []
    with open(args.scenarios) as f:
        for line in f:
            scenarios.append(json.loads(line.strip()))
    print(f"Loaded {len(scenarios)} scenarios")

    asyncio.run(generate_all(
        scenarios, skill_content, args.output,
        args.model, api_key, args.concurrency, args.resume,
    ))

if __name__ == "__main__":
    main()
