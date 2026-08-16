"""
fumii Self-Correcting Loop — Phase 2
Automated pipeline: generate → judge → rewrite → DPO → evaluate → repeat

Terminates when judge scores plateau (< 0.5 point improvement over 3 consecutive runs).

Run:
    python self_correcting_loop.py \
        --sft_checkpoint ../phase1_dataset/checkpoints/fumii-sft-v1 \
        --skill ../../shared/fumii_SKILL.md \
        --max_iterations 5 \
        --conversations_per_run 200
"""

import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Loop State
# ──────────────────────────────────────────────────────────────────────────────

class LoopState:
    def __init__(self, base_dir: str, sft_checkpoint: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.base_dir / "loop_state.json"
        self.sft_checkpoint = sft_checkpoint

        if self.state_file.exists():
            with open(self.state_file) as f:
                self._state = json.load(f)
        else:
            self._state = {
                "iteration": 0,
                "score_history": [],
                "checkpoints": [],
                "best_checkpoint": sft_checkpoint,
                "best_score": 0.0,
                "started_at": datetime.now().isoformat(),
            }

    @property
    def iteration(self) -> int:
        return self._state["iteration"]

    @property
    def score_history(self) -> list[float]:
        return self._state["score_history"]

    @property
    def best_checkpoint(self) -> str:
        return self._state["best_checkpoint"]

    def record_iteration(self, score: float, checkpoint: str):
        self._state["iteration"] += 1
        self._state["score_history"].append(score)
        self._state["checkpoints"].append(checkpoint)
        if score > self._state["best_score"]:
            self._state["best_score"] = score
            self._state["best_checkpoint"] = checkpoint
        self._save()

    def _save(self):
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def iter_dir(self) -> Path:
        d = self.base_dir / f"iteration_{self.iteration + 1:02d}"
        d.mkdir(exist_ok=True)
        return d

# ──────────────────────────────────────────────────────────────────────────────
# Termination Condition
# ──────────────────────────────────────────────────────────────────────────────

def should_continue(score_history: list[float], min_improvement: float = 0.5) -> bool:
    """
    Continue if:
    - Fewer than 3 runs completed (not enough data to judge plateau)
    - Latest 3 runs show > min_improvement in range
    """
    if len(score_history) < 3:
        return True
    recent = score_history[-3:]
    improvement = max(recent) - min(recent)
    print(f"  Recent scores: {[f'{s:.2f}' for s in recent]}")
    print(f"  Improvement range: {improvement:.2f} (threshold: {min_improvement})")
    return improvement > min_improvement

# ──────────────────────────────────────────────────────────────────────────────
# Step Runners
# ──────────────────────────────────────────────────────────────────────────────

def run_step(cmd: list[str], desc: str, cwd: str = None) -> bool:
    """Run a pipeline step as a subprocess."""
    print(f"\n── {desc} {'─' * (50 - len(desc))}")
    print(f"  CMD: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ✓ Complete ({elapsed:.0f}s)")
        return True
    else:
        print(f"  ✗ Failed (exit {result.returncode}, {elapsed:.0f}s)")
        return False

def evaluate_checkpoint(checkpoint: str, held_out_path: str, api_key: str, model: str) -> float:
    """Score a model checkpoint on the held-out set and return average judge score."""
    import asyncio
    import aiohttp
    import sys
    from fumii_pipeline.evaluation.judge_model import score_conversation

    conversations = []
    with open(held_out_path) as f:
        for line in f:
            conversations.append(json.loads(line.strip()))

    print(f"  Evaluating {len(conversations)} held-out conversations...")

    scores = []

    async def run():
        semaphore = asyncio.Semaphore(5)
        async with aiohttp.ClientSession() as session:
            tasks = [score_conversation(semaphore, session, c, api_key, model)
                     for c in conversations]
            results = await asyncio.gather(*tasks)
        for r in results:
            if r.total is not None:
                scores.append(r.total)

    asyncio.run(run())

    if not scores:
        return 0.0

    avg = sum(scores) / len(scores)
    pass_rate = sum(1 for s in scores if s >= 28) / len(scores)
    print(f"  Average score: {avg:.2f}/40")
    print(f"  Pass rate (≥28): {pass_rate:.1%}")
    return avg

# ──────────────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────────────

def run_loop(
    sft_checkpoint: str,
    skill_path: str,
    max_iterations: int,
    conversations_per_run: int,
    model: str,
    held_out_path: str,
    base_dir: str,
):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    state = LoopState(base_dir, sft_checkpoint)
    phase1_dir = str(Path(__file__).parent.parent / "phase1_dataset")
    phase2_dir = str(Path(__file__).parent)

    print(f"\n{'='*60}")
    print(f"fumii Self-Correcting Loop")
    print(f"SFT base:       {sft_checkpoint}")
    print(f"Max iterations: {max_iterations}")
    print(f"Per-run convos: {conversations_per_run}")
    print(f"{'='*60}")

    if not should_continue(state.score_history):
        print("Loop has already plateaued. Exiting.")
        return

    for i in range(state.iteration, max_iterations):
        iter_num = i + 1
        iter_dir = state.iter_dir()
        current_checkpoint = state.best_checkpoint

        print(f"\n{'='*60}")
        print(f"ITERATION {iter_num}/{max_iterations}")
        print(f"Base checkpoint: {current_checkpoint}")
        print(f"Working dir:     {iter_dir}")
        print(f"{'='*60}")

        # ── Step 1: Generate new conversations ────────────────────────────────
        scenarios_path = str(iter_dir / "scenarios.jsonl")
        generated_path = str(iter_dir / "generated.jsonl")

        ok = run_step([
            "python", str(Path(phase1_dir) / "scenario_taxonomy.py"),
            "--output", scenarios_path,
            "--count", str(conversations_per_run),
            "--seed", str(42 + iter_num),
        ], "Generating scenarios", cwd=str(iter_dir))
        if not ok:
            print("Scenario generation failed. Skipping iteration.")
            continue

        ok = run_step([
            "python", str(Path(phase1_dir) / "generate_dataset.py"),
            "--scenarios", scenarios_path,
            "--output", generated_path,
            "--skill", skill_path,
            "--model", model,
            "--concurrency", "5",
        ], "Generating conversations", cwd=str(iter_dir))
        if not ok:
            continue

        # ── Step 2: Judge ─────────────────────────────────────────────────────
        scored_path = str(iter_dir / "scored.jsonl")

        ok = run_step([
            "python", str(Path(phase1_dir) / "judge_model.py"),
            "--input", generated_path,
            "--output", scored_path,
            "--model", model,
            "--concurrency", "5",
        ], "Scoring with judge model", cwd=str(iter_dir))
        if not ok:
            continue

        # ── Step 3: Build DPO pairs ───────────────────────────────────────────
        dpo_pairs_path = str(iter_dir / "dpo_pairs.jsonl")

        ok = run_step([
            "python", str(Path(phase2_dir) / "build_dpo_pairs.py"),
            "--input", scored_path,
            "--output", dpo_pairs_path,
            "--model", model,
            "--concurrency", "5",
        ], "Building DPO pairs", cwd=str(iter_dir))
        if not ok:
            continue

        # ── Step 4: DPO Training ──────────────────────────────────────────────
        dpo_checkpoint = str(iter_dir / "dpo_checkpoint")

        ok = run_step([
            "python", str(Path(phase2_dir) / "train_dpo.py"),
            "--pairs", dpo_pairs_path,
            "--sft_checkpoint", current_checkpoint,
            "--output", dpo_checkpoint,
            "--run_name", f"dpo-iter-{iter_num}",
        ], "DPO training", cwd=str(iter_dir))
        if not ok:
            print("DPO training failed. Using previous checkpoint.")
            dpo_checkpoint = current_checkpoint

        # ── Step 5: Evaluate ──────────────────────────────────────────────────
        if held_out_path and Path(held_out_path).exists():
            print("\n── Evaluating checkpoint ──────────────────────────────────")
            avg_score = evaluate_checkpoint(dpo_checkpoint, held_out_path, api_key, model)
        else:
            print("No held-out set provided — using 0 as score estimate")
            avg_score = 0.0

        state.record_iteration(avg_score, dpo_checkpoint)

        print(f"\nIteration {iter_num} complete.")
        print(f"  Score: {avg_score:.2f}")
        print(f"  History: {[f'{s:.2f}' for s in state.score_history]}")

        # ── Termination check ─────────────────────────────────────────────────
        if not should_continue(state.score_history):
            print(f"\nPlateau detected. Stopping after iteration {iter_num}.")
            break

        if iter_num < max_iterations:
            print("Continuing to next iteration...")

    print(f"\n{'='*60}")
    print(f"Loop complete.")
    print(f"Best checkpoint: {state.best_checkpoint}")
    print(f"Best score:      {state._state['best_score']:.2f}/40")
    print(f"Score history:   {[f'{s:.2f}' for s in state.score_history]}")
    print(f"\nNext: run human spot-checks on {state.best_checkpoint}")
    print(f"Then: python ../phase1_dataset/export_gguf.py --checkpoint {state.best_checkpoint} --register")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--skill", default="../../shared/fumii_SKILL.md")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--conversations_per_run", type=int, default=200)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--held_out", default="", help="Path to held-out eval set JSONL")
    parser.add_argument("--output_dir", default="./loop_runs")
    args = parser.parse_args()

    run_loop(
        sft_checkpoint=args.sft_checkpoint,
        skill_path=args.skill,
        max_iterations=args.max_iterations,
        conversations_per_run=args.conversations_per_run,
        model=args.model,
        held_out_path=args.held_out,
        base_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
