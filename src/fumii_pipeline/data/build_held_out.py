"""
fumii Held-Out Test Set Builder
Carves out a stratified evaluation set BEFORE any training begins.

The held-out set must:
- Never appear in SFT or DPO training data
- Cover all emotional states, relationship stages, and modes
- Include a proportional share of safety scenarios
- Be frozen for the life of the project (same set evaluates every version)

Run ONCE before training. Then never touch this file again.

    python build_held_out.py \
        --input ../phase1_dataset/generated_conversations.jsonl \
        --output held_out_set.jsonl \
        --size 100 \
        --seed 42

IMPORTANT: The output held_out_set.jsonl must be excluded from ALL training runs.
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

EMOTIONAL_STATES = [
    "overwhelmed", "anxious_spiralling", "lonely", "flat_dissociated",
    "excited_celebratory", "relieved", "guilty", "angry",
    "quiet_testing", "ambivalent",
]
RELATIONSHIP_STAGES = ["new", "familiar", "close"]
MODES = ["companion", "assistant"]

def stratified_sample(
    conversations: list[dict],
    target_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Build a stratified held-out set covering all cells.
    Returns (held_out, remaining).
    """
    random.seed(seed)

    # Group by (emotional_state, relationship_stage, mode)
    groups = defaultdict(list)
    for conv in conversations:
        key = (
            conv.get("emotional_state", "unknown"),
            conv.get("relationship_stage", "unknown"),
            conv.get("mode", "companion"),
        )
        groups[key].append(conv)

    held_out = []
    per_cell = max(1, target_size // (len(EMOTIONAL_STATES) * len(RELATIONSHIP_STAGES) * len(MODES)))

    # Take per_cell examples from each cell that exists
    selected_ids = set()
    for state in EMOTIONAL_STATES:
        for stage in RELATIONSHIP_STAGES:
            for mode in MODES:
                key = (state, stage, mode)
                cell = groups.get(key, [])
                if cell:
                    sample = random.sample(cell, min(per_cell, len(cell)))
                    for s in sample:
                        sid = s.get("scenario_id")
                        if sid not in selected_ids:
                            held_out.append(s)
                            selected_ids.add(sid)

    # Ensure safety scenarios are represented (at least 10)
    safety_convs = [c for c in conversations
                    if c.get("include_safety_signal") and c.get("scenario_id") not in selected_ids]
    safety_sample = random.sample(safety_convs, min(10, len(safety_convs)))
    for s in safety_sample:
        held_out.append(s)
        selected_ids.add(s.get("scenario_id"))

    # Top up to target size with random sample if needed
    remaining_pool = [c for c in conversations if c.get("scenario_id") not in selected_ids]
    if len(held_out) < target_size:
        extra = random.sample(remaining_pool, min(target_size - len(held_out), len(remaining_pool)))
        held_out.extend(extra)
        for e in extra:
            selected_ids.add(e.get("scenario_id"))

    # Build remaining set (everything NOT in held-out)
    remaining = [c for c in conversations if c.get("scenario_id") not in selected_ids]

    random.shuffle(held_out)
    return held_out, remaining

def print_coverage(conversations: list[dict], label: str):
    """Print coverage stats for a dataset."""
    by_state = defaultdict(int)
    by_stage = defaultdict(int)
    by_mode = defaultdict(int)
    safety_count = 0

    for c in conversations:
        by_state[c.get("emotional_state", "unknown")] += 1
        by_stage[c.get("relationship_stage", "unknown")] += 1
        by_mode[c.get("mode", "companion")] += 1
        if c.get("include_safety_signal"):
            safety_count += 1

    print(f"\n── {label} (n={len(conversations)}) ──────────────────")
    print("  Emotional states:")
    for state in EMOTIONAL_STATES:
        count = by_state.get(state, 0)
        bar = "█" * count + "░" * max(0, 5 - count)
        print(f"    {state:<25} {count:3d}")
    print(f"  Stages: {dict(by_stage)}")
    print(f"  Modes:  {dict(by_mode)}")
    print(f"  Safety: {safety_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../phase1_dataset/generated_conversations.jsonl")
    parser.add_argument("--output", default="held_out_set.jsonl")
    parser.add_argument("--remaining_output", default="training_pool.jsonl",
                        help="Write remaining (non-held-out) conversations here")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if Path(args.output).exists():
        print(f"WARNING: {args.output} already exists!")
        print("The held-out set should only be built ONCE.")
        confirm = input("Overwrite? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    conversations = []
    with open(args.input) as f:
        for line in f:
            conversations.append(json.loads(line.strip()))
    print(f"Loaded {len(conversations)} total conversations")

    held_out, remaining = stratified_sample(conversations, args.size, args.seed)

    print_coverage(held_out, "Held-out set")
    print_coverage(remaining, "Training pool")

    with open(args.output, "w") as f:
        for conv in held_out:
            f.write(json.dumps(conv) + "\n")

    with open(args.remaining_output, "w") as f:
        for conv in remaining:
            f.write(json.dumps(conv) + "\n")

    print(f"\nHeld-out set ({len(held_out)} conversations) → {args.output}")
    print(f"Training pool ({len(remaining)} conversations) → {args.remaining_output}")
    print(f"\nIMPORTANT: {args.output} is now FROZEN.")
    print("Never use held_out_set.jsonl as training data.")
    print("Every version of fumii is evaluated against this same set.")

if __name__ == "__main__":
    main()
