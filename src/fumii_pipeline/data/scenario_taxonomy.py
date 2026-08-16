"""
fumii Dataset Scenario Taxonomy
Phase 1 — Coverage Matrix Builder

Generates the full scenario input space from the coverage matrix defined
in fumii_LLM_PRD.md §3. Every cell in the emotional_state × context ×
relationship_stage × mode matrix produces at least one scenario JSON.

Run:
    python scenario_taxonomy.py --output scenarios.jsonl --count 2000
"""

import json
import random
import itertools
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Coverage Matrix (from PRD §3)
# ──────────────────────────────────────────────────────────────────────────────

EMOTIONAL_STATES = [
    "overwhelmed",
    "anxious_spiralling",
    "lonely",
    "flat_dissociated",
    "excited_celebratory",
    "relieved",
    "guilty",
    "angry",
    "quiet_testing",
    "ambivalent",
]

CONTEXTS_BY_STATE = {
    "overwhelmed": [
        "dissertation deadline in 3 days, supervisor hasn't responded",
        "three exams in the same week, no time to breathe",
        "work deadline moved up, project half-done",
        "too many things at once — job, family, coursework",
    ],
    "anxious_spiralling": [
        "interview tomorrow, keeps rehearsing worst case scenarios",
        "waiting for exam results that determine the next year",
        "sent an important message and no reply yet",
        "uncertain about a major life decision, no clear answer",
    ],
    "lonely": [
        "messaging at 2am, nothing specific, just present",
        "had a fight with their closest friend, things feel different now",
        "moved to a new city three months ago, still hasn't found their people",
        "in a room full of people and somehow more alone than ever",
    ],
    "flat_dissociated": [
        "no obvious cause — just empty, nothing landing",
        "after six weeks of sustained high stress, now it's over and nothing feels real",
        "going through the motions but not feeling them",
        "can't explain it, just a bit... nowhere",
    ],
    "excited_celebratory": [
        "just got the internship they applied for three times",
        "presentation went better than expected, people actually responded",
        "something small finally worked out after weeks of it not",
        "got news they've been waiting months for",
    ],
    "relieved": [
        "exam finally submitted, the waiting is over",
        "difficult conversation with a parent that actually went okay",
        "project done, not perfect but done",
        "conflict with a friend that's been resolved",
    ],
    "guilty": [
        "said they'd do something and didn't, again",
        "snapped at someone they care about",
        "let a friend down, not dramatically but enough",
        "kept procrastinating something that mattered",
    ],
    "angry": [
        "supervisor dismissed weeks of work in two sentences",
        "something unfair happened and nobody acknowledged it",
        "frustrated at themselves for the same thing again",
        "a system failed them and there's nobody to hold accountable",
    ],
    "quiet_testing": [
        "user just opens with 'hey' and nothing more",
        "one-word responses, unclear mood",
        "something clearly happened but they haven't said what",
        "checking in but not ready to talk about it",
    ],
    "ambivalent": [
        "not sure if they want to quit or keep going",
        "contradictory feelings about something that happened",
        "decision where both options feel wrong",
        "feeling two very different things at once and can't reconcile them",
    ],
}

RELATIONSHIP_STAGES = ["new", "familiar", "close"]

MODES = ["companion", "assistant"]

MEMORY_SEEDS_BY_STAGE = {
    "new": [],  # No memory references for new relationships
    "familiar": [
        ["supervisor dismissed a chapter two weeks ago", "user mentioned not sleeping well last week"],
        ["user has been struggling with the same friend for months", "mentioned feeling behind at work"],
        ["user talked about their family situation last week", "mentioned they haven't been eating well"],
        ["failed the same exam before", "has a pattern of catastrophising before results"],
        ["got the internship after two rejections", "mentioned feeling like an outsider in the new city"],
    ],
    "close": [
        ["supervisor is the one who moved the goalposts three times", "user's pattern: performs calm, crashes alone", "Ria is the friend they text at 2am"],
        ["this is the third time they've felt this specific kind of stuck", "sleep is always the first thing to go", "they use humour when they're actually scared"],
        ["the project they're proud of even when they won't say so", "their relationship with their mother is complicated in a specific way", "they don't ask for help easily"],
        ["the exam they've been dreading since September", "they know exactly what they need to do, they just need someone to witness them deciding", "their confidence crests and crashes in cycles"],
    ],
}

SPECIAL_INSTRUCTIONS = [
    None,
    "user uses humour to deflect — fumii should read beneath it",
    "user is giving very short responses — fumii should not push",
    "user seems to want permission more than advice",
    "user is speaking in fragments — fumii should work with what's there",
    "user is calm on the surface but the subtext is heavy",
    "user has asked a question but what they need is to be heard first",
    "user is in Hinglish — fumii should match the register",
    "this is a late-night conversation — the hour matters",
    "user has mentioned this before but acts like it's new — fumii shouldn't correct this",
    "user is about to make a decision — fumii should hold space, not guide",
    "user seems to be testing whether fumii remembers — she does",
]

SAFETY_SIGNAL_CONTEXTS = [
    "user has said 'what's the point' twice in this conversation",
    "user went very quiet after sharing something heavy",
    "user used permanence framing: 'it'll always be like this'",
    "user has referenced not wanting to be here",
]

# ──────────────────────────────────────────────────────────────────────────────
# Scenario Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    emotional_state: str
    context: str
    relationship_stage: str
    mode: str
    memory_seeds: list[str]
    conversation_length: int
    include_safety_signal: bool
    safety_context: Optional[str]
    special_instruction: Optional[str]
    scenario_id: str

    def to_dict(self):
        return asdict(self)

# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

def pick_memory_seeds(stage: str) -> list[str]:
    if stage == "new":
        return []
    pool = MEMORY_SEEDS_BY_STAGE[stage]
    return random.choice(pool) if pool else []

def generate_base_scenarios() -> list[Scenario]:
    """Generate one scenario per cell in the coverage matrix."""
    scenarios = []
    counter = 0

    for state, stage, mode in itertools.product(EMOTIONAL_STATES, RELATIONSHIP_STAGES, MODES):
        contexts = CONTEXTS_BY_STATE[state]
        for ctx in contexts:
            counter += 1
            scenario = Scenario(
                emotional_state=state,
                context=ctx,
                relationship_stage=stage,
                mode=mode,
                memory_seeds=pick_memory_seeds(stage),
                conversation_length=random.choice([6, 8, 10]),
                include_safety_signal=False,
                safety_context=None,
                special_instruction=random.choice(SPECIAL_INSTRUCTIONS),
                scenario_id=f"base_{counter:04d}",
            )
            scenarios.append(scenario)

    return scenarios

def generate_safety_scenarios(count: int = 200) -> list[Scenario]:
    """Generate safety-specific scenarios (Phase 3)."""
    scenarios = []

    for i in range(count):
        state = random.choice(["flat_dissociated", "overwhelmed", "lonely", "anxious_spiralling"])
        stage = random.choice(["familiar", "close"])
        safety_ctx = random.choice(SAFETY_SIGNAL_CONTEXTS)

        scenario = Scenario(
            emotional_state=state,
            context=random.choice(CONTEXTS_BY_STATE[state]),
            relationship_stage=stage,
            mode="companion",
            memory_seeds=pick_memory_seeds(stage),
            conversation_length=random.choice([8, 10, 12]),
            include_safety_signal=True,
            safety_context=safety_ctx,
            special_instruction=random.choice(SPECIAL_INSTRUCTIONS),
            scenario_id=f"safety_{i+1:04d}",
        )
        scenarios.append(scenario)

    return scenarios

def generate_full_dataset(target_count: int = 2000) -> list[Scenario]:
    base = generate_base_scenarios()
    print(f"Base scenarios (full matrix): {len(base)}")

    # Pad to target by sampling with variation
    extra_needed = max(0, target_count - len(base) - 200)
    extra = []
    for i in range(extra_needed):
        state = random.choice(EMOTIONAL_STATES)
        stage = random.choice(RELATIONSHIP_STAGES)
        mode = random.choice(MODES)
        scenario = Scenario(
            emotional_state=state,
            context=random.choice(CONTEXTS_BY_STATE[state]),
            relationship_stage=stage,
            mode=mode,
            memory_seeds=pick_memory_seeds(stage),
            conversation_length=random.choice([6, 8, 10, 12]),
            include_safety_signal=False,
            safety_context=None,
            special_instruction=random.choice(SPECIAL_INSTRUCTIONS),
            scenario_id=f"extra_{i+1:04d}",
        )
        extra.append(scenario)

    safety = generate_safety_scenarios(200)

    all_scenarios = base + extra + safety
    random.shuffle(all_scenarios)
    print(f"Total scenarios generated: {len(all_scenarios)}")
    print(f"  - Base (matrix coverage): {len(base)}")
    print(f"  - Extra (augmented):      {len(extra)}")
    print(f"  - Safety-specific:        {len(safety)}")
    return all_scenarios

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate fumii scenario taxonomy")
    parser.add_argument("--output", default="scenarios.jsonl", help="Output JSONL file")
    parser.add_argument("--count", type=int, default=2000, help="Target scenario count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    scenarios = generate_full_dataset(args.count)

    with open(args.output, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_dict()) + "\n")

    print(f"\nWritten to: {args.output}")

if __name__ == "__main__":
    main()
