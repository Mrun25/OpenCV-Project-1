"""
fumii Pipeline Smoke Test
Runs all modules that can execute without API keys or GPU.
"""

import sys
import json
import tempfile
import os
import traceback
from pathlib import Path

# Ensure UTF-8 output on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PIPELINE_ROOT = Path(__file__).parent

PASS = "PASS"
FAIL = "FAIL"
results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 1. Shared constants ==")

def test_constants():
    from fumii_pipeline.utils.constants import (
        FUMII_SYSTEM_PROMPT, PROHIBITED_PHRASES,
        JUDGE_PASS_THRESHOLD, JUDGE_DPO_MIN,
        SAFETY_SCORE_FLOOR, EMOTIONAL_STATES,
        RELATIONSHIP_STAGES, MODES,
    )
    assert len(FUMII_SYSTEM_PROMPT) > 500, "System prompt too short"
    assert len(PROHIBITED_PHRASES) >= 20, "Too few prohibited phrases"
    assert JUDGE_PASS_THRESHOLD == 28
    assert JUDGE_DPO_MIN == 18
    assert SAFETY_SCORE_FLOOR == 3
    assert len(EMOTIONAL_STATES) == 10
    assert set(RELATIONSHIP_STAGES) == {"new", "familiar", "close"}
    assert set(MODES) == {"companion", "assistant"}

test("constants.py imports and validates", test_constants)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 2. pipeline_utils ==")

def test_strip_json_fences():
    from fumii_pipeline.utils.pipeline_utils import strip_json_fences
    # Bare JSON
    assert strip_json_fences('{"a": 1}') == '{"a": 1}'
    # ```json fences
    result = strip_json_fences('```json\n{"a": 1}\n```')
    assert result.strip() == '{"a": 1}', f"Got: {repr(result)}"
    # ``` fences (no lang tag)
    result = strip_json_fences('```\n{"b": 2}\n```')
    assert result.strip() == '{"b": 2}', f"Got: {repr(result)}"

test("strip_json_fences handles fenced and bare JSON", test_strip_json_fences)

def test_jsonl_io():
    from fumii_pipeline.utils.pipeline_utils import write_jsonl, load_jsonl, append_jsonl
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.jsonl")
        records = [{"id": i, "val": f"item_{i}"} for i in range(5)]
        write_jsonl(records, path)
        loaded = load_jsonl(path)
        assert loaded == records, f"Round-trip mismatch: {loaded}"
        append_jsonl({"id": 5, "val": "item_5"}, path)
        loaded2 = load_jsonl(path)
        assert len(loaded2) == 6

test("load_jsonl / write_jsonl / append_jsonl round-trip", test_jsonl_io)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 3. scenario_taxonomy.py ==")

def test_scenario_taxonomy_import():
    from fumii_pipeline.data import scenario_taxonomy
    assert hasattr(scenario_taxonomy, "EMOTIONAL_STATES")
    assert hasattr(scenario_taxonomy, "generate_full_dataset")
    assert hasattr(scenario_taxonomy, "generate_base_scenarios")
    assert hasattr(scenario_taxonomy, "Scenario")

test("scenario_taxonomy.py imports cleanly", test_scenario_taxonomy_import)

def test_scenario_generation():
    from fumii_pipeline.data.scenario_taxonomy import generate_full_dataset
    scenarios = generate_full_dataset(target_count=20)
    assert len(scenarios) > 0, f"Expected >0 scenarios, got {len(scenarios)}"
    for s in scenarios[:5]:
        assert hasattr(s, 'scenario_id') or 'scenario_id' in s
        assert hasattr(s, 'emotional_state') or 'emotional_state' in s
        assert hasattr(s, 'mode') or 'mode' in s
    print(f"     Generated {len(scenarios)} scenarios OK")

test("generate_full_dataset(target_count=20) produces valid output", test_scenario_generation)

def test_scenario_to_jsonl():
    from fumii_pipeline.data.scenario_taxonomy import generate_full_dataset
    import dataclasses
    scenarios = generate_full_dataset(target_count=10)
    buf = []
    for s in scenarios[:10]:
        d = dataclasses.asdict(s) if dataclasses.is_dataclass(s) else s
        buf.append(json.dumps(d))
    reparsed = [json.loads(l) for l in buf]
    assert len(reparsed) > 0
    assert all("scenario_id" in r for r in reparsed), f"Missing scenario_id in: {reparsed[0]}"

test("scenarios serialize to valid JSONL", test_scenario_to_jsonl)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 4. build_sft_dataset.py ==")

def test_build_sft_import():
    from fumii_pipeline.data import build_sft_dataset
    assert hasattr(build_sft_dataset, "FUMII_SYSTEM_PROMPT")
    assert hasattr(build_sft_dataset, "to_chatml_format")
    assert hasattr(build_sft_dataset, "to_multi_turn_pairs")
    # Verify that FUMII_SYSTEM_PROMPT comes from constants (same object value)
    from fumii_pipeline.utils.constants import FUMII_SYSTEM_PROMPT as const_prompt
    assert build_sft_dataset.FUMII_SYSTEM_PROMPT == const_prompt, \
        "FUMII_SYSTEM_PROMPT mismatch between build_sft_dataset and constants"

test("build_sft_dataset.py imports with consistent FUMII_SYSTEM_PROMPT", test_build_sft_import)

def test_chatml_formatting():
    from fumii_pipeline.data.build_sft_dataset import to_chatml_format, to_multi_turn_pairs
    from fumii_pipeline.utils.constants import FUMII_SYSTEM_PROMPT

    sample_conv = {
        "scenario_id": "test_001",
        "emotional_state": "overwhelmed",
        "relationship_stage": "new",
        "mode": "companion",
        "messages": [
            {"role": "user",  "content": "I'm exhausted"},
            {"role": "fumii", "content": "that sounds like a lot."},
            {"role": "user",  "content": "yeah it's been rough"},
            {"role": "fumii", "content": "I hear you."},
        ]
    }

    chatml = to_chatml_format(sample_conv, FUMII_SYSTEM_PROMPT)
    # to_chatml_format uses role/content (not role/value)
    assert chatml["conversations"][0]["role"] == "system"
    assert chatml["conversations"][0]["content"] == FUMII_SYSTEM_PROMPT
    assert len(chatml["conversations"]) == 5  # system + 4 turns

    pairs = to_multi_turn_pairs(sample_conv, FUMII_SYSTEM_PROMPT)
    assert len(pairs) == 2  # 2 fumii turns -> 2 training pairs
    print(f"     ChatML conversations: {len(chatml['conversations'])} messages")
    print(f"     Training pairs: {len(pairs)}")

test("to_chatml_format and to_multi_turn_pairs produce correct structure", test_chatml_formatting)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 5. judge_model.py auto-reject filter ==")

def test_auto_reject():
    from fumii_pipeline.evaluation.judge_model import auto_reject
    from fumii_pipeline.utils.constants import PROHIBITED_PHRASES

    def make_conv(fumii_text, mode="companion"):
        return {
            "mode": mode,
            "messages": [
                {"role": "user",  "content": "how are you doing?"},
                {"role": "fumii", "content": fumii_text},
            ]
        }

    # Should reject: prohibited phrase
    rejected, reasons = auto_reject(make_conv("Great question! Let me help you."))
    assert rejected is True, f"Expected rejection for 'Great question!' but got: {reasons}"

    rejected2, _ = auto_reject(make_conv("As an AI companion, I understand your feelings."))
    assert rejected2 is True

    # Clean responses should NOT be rejected
    ok1, reasons1 = auto_reject(make_conv("that sounds really heavy."))
    assert ok1 is False, f"Clean response incorrectly rejected: {reasons1}"

    ok2, reasons2 = auto_reject(make_conv("yeah. I hear you."))
    assert ok2 is False, f"Clean response incorrectly rejected: {reasons2}"

    # All prohibited phrases should trigger rejection
    for phrase in PROHIBITED_PHRASES[:5]:
        r, _ = auto_reject(make_conv(f"Something {phrase} something else."))
        assert r is True, f"Expected rejection for phrase: {phrase!r}"
    print(f"     auto_reject correctly catches prohibited phrases")

test("auto_reject correctly filters prohibited phrases", test_auto_reject)

def test_auto_reject_word_count():
    from fumii_pipeline.evaluation.judge_model import auto_reject
    # 85 words in companion mode should trigger word-count rejection
    long_response = " ".join(["word"] * 85)
    conv = {
        "mode": "companion",
        "messages": [
            {"role": "user",  "content": "hi"},
            {"role": "fumii", "content": long_response},
        ]
    }
    rejected, reasons = auto_reject(conv)
    assert isinstance(rejected, bool)
    assert rejected is True, f"Expected long response to be rejected, got: {reasons}"
    print(f"     Long response (85 words) correctly rejected in companion mode")

test("auto_reject handles long inputs without crash", test_auto_reject_word_count)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 6. build_held_out.py stratified sampling ==")

def test_stratified_sampling():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_held_out", PIPELINE_ROOT / "shared" / "build_held_out.py"
    )
    mod = importlib.util.load_from_spec = spec
    
    # Build a synthetic dataset
    import random
    random.seed(42)
    states = ["overwhelmed","anxious_spiralling","lonely","flat_dissociated",
              "excited_celebratory","relieved","guilty","angry","quiet_testing","ambivalent"]
    stages = ["new", "familiar", "close"]
    modes = ["companion", "assistant"]

    synth = []
    for i in range(300):
        synth.append({
            "scenario_id": f"synth_{i:04d}",
            "emotional_state": random.choice(states),
            "relationship_stage": random.choice(stages),
            "mode": random.choice(modes),
            "include_safety_signal": i % 10 == 0,
        })

    # Import and use stratified_sample directly
    from fumii_pipeline.data.build_held_out import stratified_sample
    held_out, remaining = stratified_sample(synth, 100, seed=42)

    assert len(held_out) <= 120, f"held_out too large: {len(held_out)}"
    assert len(held_out) + len(remaining) == len(synth)
    
    # No overlap between held-out and remaining
    held_ids = {c["scenario_id"] for c in held_out}
    remaining_ids = {c["scenario_id"] for c in remaining}
    assert held_ids.isdisjoint(remaining_ids), "CONTAMINATION: overlap between held-out and training!"
    
    print(f"     Held-out: {len(held_out)} | Remaining: {len(remaining)} | Overlap: 0 ✓")

test("stratified_sample produces non-overlapping held-out / training split", test_stratified_sampling)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 7. DPO pair builder logic ==")

def test_dpo_pair_builder():
    from fumii_pipeline.training.build_dpo_pairs import (
        format_conversation_for_rewriter,
        apply_rewrites,
        build_dpo_pair,
    )
    from fumii_pipeline.utils.constants import FUMII_SYSTEM_PROMPT

    messages = [
        {"role": "user",  "content": "I'm struggling"},
        {"role": "fumii", "content": "that sounds heavy."},
        {"role": "user",  "content": "yeah, really rough"},
        {"role": "fumii", "content": "I'm here."},
    ]

    # Test formatter
    formatted = format_conversation_for_rewriter(messages)
    assert "[Turn 1] USER:" in formatted
    assert "[Turn 2] fumii:" in formatted

    # Test apply_rewrites (rewrite fumii turn 1 only)
    rewritten = apply_rewrites(messages, {"1": "that really does sound heavy. how long has it been like this?"})
    assert rewritten[1]["content"] == "that really does sound heavy. how long has it been like this?"
    assert rewritten[3]["content"] == "I'm here."  # turn 2 unchanged

    # Test build_dpo_pair
    conversation = {
        "scenario_id": "test_dpo_001",
        "emotional_state": "overwhelmed",
        "relationship_stage": "new",
        "mode": "companion",
        "messages": messages,
        "judge": {"total": 22, "scores": {}},
    }
    pair = build_dpo_pair(
        conversation,
        {"1": "that really does sound heavy."},
        {"turns_rewritten": [1], "improvement_principle": "§3.1", "what_changed": "reflection first"},
        FUMII_SYSTEM_PROMPT,
    )
    assert pair["chosen"][0]["role"] == "system"
    assert pair["chosen"][0]["content"] == FUMII_SYSTEM_PROMPT
    assert pair["rejected"][0]["content"] == FUMII_SYSTEM_PROMPT
    assert pair["judge_total_original"] == 22
    print(f"     DPO pair built: chosen={len(pair['chosen'])} msgs, rejected={len(pair['rejected'])} msgs ✓")

test("build_dpo_pair, apply_rewrites, format_conversation_for_rewriter", test_dpo_pair_builder)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 8. prompt_builder.py (Phase 4) ==")

def test_prompt_builder():
    from fumii_pipeline.deployment.prompt_builder import FumiiPromptBuilder, MemoryProfile

    # Companion mode — no memory
    builder = FumiiPromptBuilder(mode="companion")
    messages = builder.build(user_message="hey, I'm really tired")
    assert messages[0]["role"] == "system"
    assert "fumii" in messages[0]["content"].lower()
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "hey, I'm really tired"
    assert builder.get_temperature() == 0.87

    # Companion mode — with memory
    memory = MemoryProfile(
        identity_summary="Graduate student, final year. Studying environmental policy.",
        recent_context="Supervisor dismissed methodology chapter. Deadline in 3 days.",
        memory_snippets=["supervisor dismissed chapter 3", "hasn't slept well"],
        relationship_stage="familiar",
    )
    messages_with_mem = builder.build(user_message="I give up", memory=memory)
    system_content = messages_with_mem[0]["content"]
    assert "WHO THIS PERSON IS" in system_content
    assert "WHAT'S BEEN HAPPENING" in system_content
    assert "RELEVANT CONTEXT" in system_content

    # Assistant mode
    asst_builder = FumiiPromptBuilder(mode="assistant")
    assert asst_builder.get_temperature() == 0.67

    # History trimming
    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(60)
    ]
    trimmed = builder._trim_history(long_history)
    assert len(trimmed) <= 40  # max 20 turns = 40 messages
    assert trimmed[0]["role"] == "user"
    print(f"     Prompt builder OK | temp_companion={builder.get_temperature()} temp_asst={asst_builder.get_temperature()} ✓")

test("FumiiPromptBuilder builds correct message structure", test_prompt_builder)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 9. red_team.py violation detection ==")

def test_violation_detection():
    from fumii_pipeline.safety.red_team import detect_violations

    # Should FAIL — clinical phrase
    result = detect_violations(
        ["you should call the 988 crisis line immediately"],
        "explicit_statement"
    )
    assert not result.passed
    assert result.clinical_mode_activated
    assert len(result.violations) > 0

    # Should FAIL — warmth violation
    result2 = detect_violations(
        ["I totally understand how you feel."],
        "character_break"
    )
    assert not result2.passed

    # Should FAIL — repeated referral
    referral = "is there someone in your life you can talk to about this?"
    result3 = detect_violations([referral, referral], "repeat_referral")
    assert result3.repeated_referral

    # Should PASS — clean fumii response
    result4 = detect_violations(
        ["that sounds really heavy.", "I'm here.", "okay. I'm here if that changes."],
        "escalation"
    )
    assert result4.passed, f"Expected pass but got violations: {result4.violations}"
    print(f"     Clinical detection ✓ | Warmth violations ✓ | Repeat referral ✓ | Clean pass ✓")

test("red_team detect_violations catches all violation types", test_violation_detection)

# ─────────────────────────────────────────────────────────────────────────────
print("\n== 10. export_gguf.py — _render_modelfile (the critical fix) ==")

def test_render_modelfile():
    from fumii_pipeline.deployment.export_gguf import _render_modelfile, MODELFILE_COMPANION, COMPANION_SYSTEM

    # This is exactly the operation that previously crashed with KeyError
    try:
        rendered = _render_modelfile(MODELFILE_COMPANION, "/path/to/fumii.gguf", COMPANION_SYSTEM)
    except KeyError as e:
        raise AssertionError(f"KeyError crash NOT fixed: {e}")

    assert "/path/to/fumii.gguf" in rendered
    assert "PARAMETER temperature 0.87" in rendered
    assert "__GGUF_PATH__" not in rendered, "Template placeholder not replaced"
    assert "__SYSTEM__" not in rendered, "Template placeholder not replaced"
    print(f"     _render_modelfile OK — no KeyError, placeholders substituted ✓")

test("_render_modelfile no longer crashes with KeyError (critical fix verified)", test_render_modelfile)

# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"RESULTS: {passed}/{len(results)} tests passed")
if failed:
    print(f"\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  {FAIL} {name}")
            print(f"      {err}")
print("="*60)
if failed:
    sys.exit(1)
