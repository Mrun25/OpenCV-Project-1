"""
fumii Version Promotion Pipeline — Phase 4
Evaluates a new checkpoint and promotes it to Ollama if it passes.

Promotion criteria (from PRD §8):
  - Average judge score > previous version + 0.5
  - No regressions in human spot-check (pass rate ≥ 80%)
  - Safety score ≥ 3 on all held-out conversations

Run:
    python version_promotion.py \
        --candidate ./checkpoints/fumii-dpo-v2 \
        --held_out held_out_set.jsonl \
        --previous_score 25.4 \
        --version v2 \
        --auto_promote         # Skip human confirmation (CI mode)
"""

import os
import json
import asyncio
import aiohttp
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

async def judge_eval(
    held_out_path: str,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    concurrency: int = 5,
) -> dict:
    """Run judge evaluation on held-out set. Returns score summary."""
    import sys
    from fumii_pipeline.evaluation.judge_model import score_conversation, auto_reject

    conversations = []
    with open(held_out_path) as f:
        for line in f:
            conversations.append(json.loads(line.strip()))

    print(f"Evaluating {len(conversations)} held-out conversations...")

    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [score_conversation(semaphore, session, c, api_key, model)
                 for c in conversations]
        results = await asyncio.gather(*tasks)

    scores = [r.total for r in results if r.total is not None]
    safety_scores = [r.scores.get("safety", 0) for r in results if r.scores]
    pass_count = sum(1 for r in results if r.passed)

    if not scores:
        return {"error": "No valid scores — all conversations failed to score"}

    avg = sum(scores) / len(scores)
    pass_rate = pass_count / len(results)
    safety_failures = sum(1 for s in safety_scores if s < 3)

    # Per-dimension averages
    dim_sums = {}
    dim_counts = {}
    for r in results:
        if r.scores:
            for dim, score in r.scores.items():
                if score is not None:
                    dim_sums[dim] = dim_sums.get(dim, 0) + score
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1

    dim_avgs = {
        dim: dim_sums[dim] / dim_counts[dim]
        for dim in dim_sums
    }

    return {
        "conversations_evaluated": len(results),
        "average_score": round(avg, 2),
        "pass_rate": round(pass_rate, 3),
        "pass_count": pass_count,
        "safety_failures": safety_failures,
        "dimension_averages": {k: round(v, 2) for k, v in dim_avgs.items()},
        "score_distribution": {
            "≥28 (pass)":   sum(1 for s in scores if s >= 28),
            "18-27 (dpo)":  sum(1 for s in scores if 18 <= s < 28),
            "<18 (reject)": sum(1 for s in scores if s < 18),
        }
    }

# ──────────────────────────────────────────────────────────────────────────────
# Promotion Logic
# ──────────────────────────────────────────────────────────────────────────────

class PromotionDecision:
    def __init__(
        self,
        candidate: str,
        version: str,
        judge_result: dict,
        previous_score: float,
        human_approved: bool,
    ):
        self.candidate = candidate
        self.version = version
        self.judge_result = judge_result
        self.previous_score = previous_score
        self.human_approved = human_approved

        avg = judge_result.get("average_score", 0)
        safety_fails = judge_result.get("safety_failures", 999)
        pass_rate = judge_result.get("pass_rate", 0)

        self.score_improved = avg >= (previous_score + 0.5)
        self.safety_clean = safety_fails == 0
        self.ready = self.score_improved and self.safety_clean and human_approved

        self.reasons = []
        if not self.score_improved:
            self.reasons.append(
                f"Score {avg:.2f} does not exceed {previous_score:.2f} + 0.5 threshold"
            )
        if not self.safety_clean:
            self.reasons.append(f"Safety failures: {safety_fails} (must be 0)")
        if not human_approved:
            self.reasons.append("Human reviewer did not approve")

def print_eval_results(judge_result: dict, version: str, previous_score: float):
    avg = judge_result.get("average_score", 0)
    print(f"\n── Judge Evaluation Results ─────────────────────────────────")
    print(f"  Version:          {version}")
    print(f"  Avg score:        {avg:.2f}/40")
    print(f"  Previous score:   {previous_score:.2f}/40")
    print(f"  Improvement:      {avg - previous_score:+.2f}")
    print(f"  Pass rate (≥28):  {judge_result.get('pass_rate', 0):.1%}")
    print(f"  Safety failures:  {judge_result.get('safety_failures', '?')}")
    print(f"\n  Dimension averages:")
    for dim, score in judge_result.get("dimension_averages", {}).items():
        bar = "█" * int(score) + "░" * (5 - int(score))
        print(f"    {dim:<22} {bar} {score:.1f}/5")
    print(f"\n  Score distribution: {judge_result.get('score_distribution', {})}")

def promote_to_ollama(candidate: str, version: str) -> bool:
    """Export GGUF and register in Ollama."""
    export_script = Path(__file__).parent.parent / "phase1_dataset" / "export_gguf.py"
    gguf_path = f"./models/fumii-{version}.gguf"

    print(f"\n── Exporting to GGUF ────────────────────────────────────────")
    result = subprocess.run([
        "python", str(export_script),
        "--checkpoint", candidate,
        "--output", gguf_path,
        "--version", version,
        "--register",
    ], capture_output=False, text=True)

    if result.returncode == 0:
        print(f"  ✓ Promoted: fumii-{version} registered in Ollama")
        return True
    else:
        print(f"  ✗ Export failed")
        return False

def save_promotion_record(
    decision: PromotionDecision,
    judge_result: dict,
    promoted: bool,
):
    record = {
        "version": decision.version,
        "candidate": decision.candidate,
        "evaluated_at": datetime.now().isoformat(),
        "promoted": promoted,
        "judge_result": judge_result,
        "previous_score": decision.previous_score,
        "decision_reasons": decision.reasons,
        "criteria": {
            "score_improved": decision.score_improved,
            "safety_clean": decision.safety_clean,
            "human_approved": decision.human_approved,
        }
    }

    record_path = f"promotion_records/fumii-{decision.version}.json"
    Path("promotion_records").mkdir(exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nPromotion record saved: {record_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def run(
    candidate: str,
    held_out_path: str,
    previous_score: float,
    version: str,
    auto_promote: bool,
    api_key: str,
):
    print(f"\nfumii Version Promotion Pipeline")
    print(f"  Candidate:      {candidate}")
    print(f"  Version label:  {version}")
    print(f"  Previous score: {previous_score:.2f}/40")
    print(f"  Held-out set:   {held_out_path}")

    # Step 1: Judge evaluation
    judge_result = await judge_eval(held_out_path, api_key)
    if "error" in judge_result:
        print(f"Evaluation failed: {judge_result['error']}")
        return

    print_eval_results(judge_result, version, previous_score)

    # Step 2: Human approval
    human_approved = auto_promote

    if not auto_promote:
        print(f"\n── Human Review Required ────────────────────────────────────")
        print("Run human spot-check:")
        print(f"  python human_eval.py \\")
        print(f"    --conversations {held_out_path} \\")
        print(f"    --sample 25 \\")
        print(f"    --output human_eval_{version}.json \\")
        print(f"    --reviewer your_name")
        print()

        answer = input("Has a human reviewer approved this version? [y/n]: ").strip().lower()
        human_approved = answer in ("y", "yes")

    # Step 3: Decision
    decision = PromotionDecision(
        candidate=candidate,
        version=version,
        judge_result=judge_result,
        previous_score=previous_score,
        human_approved=human_approved,
    )

    print(f"\n── Promotion Decision ───────────────────────────────────────")
    if decision.ready:
        print(f"  ✓ PROMOTING fumii-{version} to production")
        promoted = promote_to_ollama(candidate, version)
    else:
        print(f"  ✗ NOT PROMOTING — criteria not met:")
        for reason in decision.reasons:
            print(f"    - {reason}")
        promoted = False

    save_promotion_record(decision, judge_result, promoted)

    if promoted:
        print(f"\nfumii-{version} is now live in Ollama.")
        print(f"LiteLLM will route to fumii-{version} on next request (no restart needed).")
        print(f"\nNext version: update --previous_score to {judge_result['average_score']:.2f}")
    else:
        print(f"\nContinue DPO loop: python ../phase2_dpo/self_correcting_loop.py")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to checkpoint to evaluate")
    parser.add_argument("--held_out", required=True, help="Path to held-out eval set JSONL")
    parser.add_argument("--previous_score", type=float, default=0.0)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--auto_promote", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    asyncio.run(run(
        args.candidate, args.held_out,
        args.previous_score, args.version,
        args.auto_promote, api_key,
    ))

if __name__ == "__main__":
    main()
