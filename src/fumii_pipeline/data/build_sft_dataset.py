"""
fumii SFT Dataset Builder — Phase 1
Converts judge-scored conversations into the format required for SFT training.

Reads scored_conversations.jsonl, filters for routing == "sft",
and writes Alpaca-style and ChatML-style JSONL for Axolotl / Unsloth.

Run:
    python build_sft_dataset.py \
        --input scored_conversations.jsonl \
        --output_alpaca sft_alpaca.jsonl \
        --output_chatml sft_chatml.jsonl \
        --skill ../../shared/fumii_SKILL.md
"""

import json
import sys
import argparse
from pathlib import Path

from fumii_pipeline.utils.constants import FUMII_SYSTEM_PROMPT  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Format Converters
# ──────────────────────────────────────────────────────────────────────────────

def to_alpaca_format(conversation: dict, system_prompt: str) -> dict:
    """
    Alpaca format: single instruction/output pair per conversation.
    The full conversation history is encoded as the instruction.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return None

    # Build conversation history as instruction context
    history_lines = []
    last_fumii_response = None

    for msg in messages:
        if msg["role"] == "user":
            history_lines.append(f"User: {msg['content']}")
        else:
            if history_lines:  # There's at least one user message before this
                last_fumii_response = msg["content"]
            history_lines.append(f"fumii: {msg['content']}")

    # Last fumii turn is the output; everything before is the instruction
    if not last_fumii_response:
        return None

    # Find the last fumii response and use everything before it as instruction
    last_fumii_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "fumii":
            last_fumii_idx = i
            break

    if last_fumii_idx is None or last_fumii_idx == 0:
        return None

    context_messages = messages[:last_fumii_idx]
    context_lines = []
    for msg in context_messages:
        role = "User" if msg["role"] == "user" else "fumii"
        context_lines.append(f"{role}: {msg['content']}")

    instruction = "\n\n".join(context_lines)
    output = messages[last_fumii_idx]["content"]

    return {
        "system": system_prompt,
        "instruction": instruction,
        "input": "",
        "output": output,
        "metadata": {
            "scenario_id": conversation.get("scenario_id"),
            "emotional_state": conversation.get("emotional_state"),
            "relationship_stage": conversation.get("relationship_stage"),
            "mode": conversation.get("mode"),
            "judge_total": conversation.get("judge", {}).get("total"),
        }
    }

def to_chatml_format(conversation: dict, system_prompt: str) -> dict:
    """
    ChatML format: full multi-turn conversation for Axolotl sharegpt training.
    This is the preferred format — it trains on every fumii turn, not just the last.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return None

    chatml_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        chatml_messages.append({"role": role, "content": msg["content"]})

    return {
        "conversations": chatml_messages,
        "metadata": {
            "scenario_id": conversation.get("scenario_id"),
            "emotional_state": conversation.get("emotional_state"),
            "relationship_stage": conversation.get("relationship_stage"),
            "mode": conversation.get("mode"),
            "judge_total": conversation.get("judge", {}).get("total"),
        }
    }

def to_multi_turn_pairs(conversation: dict, system_prompt: str) -> list[dict]:
    """
    Unsloth multi-turn format: generates one training example per fumii turn.
    Maximises the number of training examples from each conversation.
    """
    messages = conversation.get("messages", [])
    pairs = []

    for i, msg in enumerate(messages):
        if msg["role"] != "fumii":
            continue

        # Build conversation up to and including this fumii turn
        window = messages[:i + 1]
        chatml = [{"role": "system", "content": system_prompt}]
        for m in window:
            role = "user" if m["role"] == "user" else "assistant"
            chatml.append({"role": role, "content": m["content"]})

        pairs.append({
            "conversations": chatml,
            "metadata": {
                "scenario_id": conversation.get("scenario_id"),
                "turn_index": i,
                "emotional_state": conversation.get("emotional_state"),
                "relationship_stage": conversation.get("relationship_stage"),
                "mode": conversation.get("mode"),
            }
        })

    return pairs

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build SFT training dataset")
    parser.add_argument("--input", default="scored_conversations.jsonl")
    parser.add_argument("--output_alpaca", default="sft_alpaca.jsonl")
    parser.add_argument("--output_chatml", default="sft_chatml.jsonl")
    parser.add_argument("--output_pairs", default="sft_pairs.jsonl")
    parser.add_argument("--format", choices=["alpaca", "chatml", "pairs", "all"], default="all")
    parser.add_argument("--min_score", type=int, default=28, help="Minimum judge total to include")
    args = parser.parse_args()

    system_prompt = FUMII_SYSTEM_PROMPT

    # Load and filter SFT-routed conversations
    sft_conversations = []
    total_loaded = 0
    with open(args.input) as f:
        for line in f:
            total_loaded += 1
            obj = json.loads(line.strip())
            judge = obj.get("judge", {})
            if (
                judge.get("routing") == "sft"
                and (judge.get("total") or 0) >= args.min_score
                and (judge.get("scores") or {}).get("safety", 0) >= 3
            ):
                sft_conversations.append(obj)

    print(f"Loaded {total_loaded} total conversations")
    print(f"SFT-approved (score ≥ {args.min_score}, safety ≥ 3): {len(sft_conversations)}")

    if not sft_conversations:
        print("No SFT conversations found. Check judge routing in scored_conversations.jsonl.")
        return

    # Stats breakdown
    by_state = {}
    by_stage = {}
    by_mode = {}
    for conv in sft_conversations:
        s = conv.get("emotional_state", "unknown")
        st = conv.get("relationship_stage", "unknown")
        m = conv.get("mode", "unknown")
        by_state[s] = by_state.get(s, 0) + 1
        by_stage[st] = by_stage.get(st, 0) + 1
        by_mode[m] = by_mode.get(m, 0) + 1

    print("\nCoverage breakdown:")
    print("  Emotional states:")
    for k, v in sorted(by_state.items()):
        print(f"    {k}: {v}")
    print("  Relationship stages:", by_stage)
    print("  Modes:", by_mode)

    # Write outputs
    if args.format in ("alpaca", "all"):
        count = 0
        with open(args.output_alpaca, "w") as f:
            for conv in sft_conversations:
                record = to_alpaca_format(conv, system_prompt)
                if record:
                    f.write(json.dumps(record) + "\n")
                    count += 1
        print(f"\nAlpaca format: {count} examples → {args.output_alpaca}")

    if args.format in ("chatml", "all"):
        count = 0
        with open(args.output_chatml, "w") as f:
            for conv in sft_conversations:
                record = to_chatml_format(conv, system_prompt)
                if record:
                    f.write(json.dumps(record) + "\n")
                    count += 1
        print(f"ChatML format: {count} conversations → {args.output_chatml}")

    if args.format in ("pairs", "all"):
        count = 0
        with open(args.output_pairs, "w") as f:
            for conv in sft_conversations:
                pairs = to_multi_turn_pairs(conv, system_prompt)
                for p in pairs:
                    f.write(json.dumps(p) + "\n")
                    count += 1
        print(f"Multi-turn pairs format: {count} examples → {args.output_pairs}")

if __name__ == "__main__":
    main()
