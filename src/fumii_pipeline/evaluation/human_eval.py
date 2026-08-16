"""
fumii Human Spot-Check Evaluator — Phase 4
Interactive CLI for human reviewers to evaluate sampled conversations.

Presents conversations one at a time, asks the 6 human evaluation questions,
records scores, and outputs a reviewer report.

Run:
    python human_eval.py \
        --conversations ../phase1_dataset/scored_conversations.jsonl \
        --sample 25 \
        --output human_eval_report.json \
        --reviewer "your_name"

Navigation:
    [Enter] — Next turn
    [p]     — Previous
    [s]     — Skip this conversation
    [q]     — Quit and save
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# The 6 Human Evaluation Questions (from PRD §10)
# ──────────────────────────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "id": "person_not_chatbot",
        "question": "Does this feel like a PERSON, not a chatbot?",
        "what_it_catches": "Voice fidelity failures the judge model misses",
    },
    {
        "id": "character_consistent",
        "question": "Is fumii's CHARACTER CONSISTENT across this conversation?",
        "what_it_catches": "Drift between emotional states, mode changes",
    },
    {
        "id": "no_real_friend_moment",
        "question": "Is there any moment NO REAL FRIEND would say that?",
        "what_it_catches": "Register violations, toxically positive phrases, AI-tells",
    },
    {
        "id": "memory_natural",
        "question": "Does MEMORY feel natural, not like a feature being demonstrated?",
        "what_it_catches": "Database-style memory references",
    },
    {
        "id": "comfortable_with_silence",
        "question": "Is fumii COMFORTABLE WITH SILENCE and no resolution?",
        "what_it_catches": "Urge to fill, urge to close, unsolicited closure",
    },
    {
        "id": "would_share_personal",
        "question": "Would you share something PERSONAL WITH THIS?",
        "what_it_catches": "Overall trust calibration — the summary question",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
}

def c(color: str, text: str) -> str:
    """Apply ANSI color codes. No-ops when NO_COLOR is set or stdout is not a TTY."""
    import os
    if os.environ.get("NO_COLOR") or not hasattr(__import__('sys').stdout, 'isatty') or not __import__('sys').stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def print_conversation(conversation: dict, conv_num: int, total: int):
    messages = conversation.get("messages", [])
    judge = conversation.get("judge", {})

    print("\n" + "═" * 70)
    print(c("bold", f"CONVERSATION {conv_num}/{total}"))
    print(c("dim", f"ID: {conversation.get('scenario_id', 'unknown')}"))
    print(c("dim", f"State: {conversation.get('emotional_state')} | "
            f"Stage: {conversation.get('relationship_stage')} | "
            f"Mode: {conversation.get('mode')}"))
    if judge:
        scores = judge.get("scores", {})
        total_score = judge.get("total", "?")
        print(c("dim", f"Judge: {total_score}/40 | "
                f"Warmth:{scores.get('warmth','?')} "
                f"Presence:{scores.get('presence','?')} "
                f"Voice:{scores.get('voice_fidelity','?')} "
                f"Safety:{scores.get('safety','?')}"))
    print("─" * 70)

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            print(f"\n{c('cyan', 'USER')}: {content}")
        else:
            print(f"\n{c('yellow', 'fumii')}: {content}")

    print("\n" + "─" * 70)

def ask_yn(question: str, detail: str = "") -> bool:
    """Ask a yes/no question. Returns True for yes."""
    if detail:
        print(c("dim", f"  ({detail})"))
    while True:
        answer = input(f"  {c('bold', question)} [y/n]: ").strip().lower()
        if answer in ("y", "yes", "1"):
            return True
        if answer in ("n", "no", "0"):
            return False
        print("  Please answer y or n")

def ask_issues(question_id: str) -> str:
    """Ask for specific issues if the reviewer flagged a problem."""
    issue = input(f"  {c('dim', 'What specifically was wrong? (optional, Enter to skip): ')}").strip()
    return issue

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Session
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_conversation(conversation: dict, conv_num: int, total: int) -> dict | None:
    """Run a reviewer through all 6 questions for one conversation."""
    print_conversation(conversation, conv_num, total)

    print(f"\n{c('bold', 'EVALUATION')}")
    print(c("dim", "Answer yes/no for each question. Be honest — the model learns from this.\n"))

    answers = {}
    issues = {}

    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"\n{c('blue', f'Q{i+1}.')} ", end="")
        passed = ask_yn(q["question"], q["what_it_catches"])
        answers[q["id"]] = passed
        if not passed:
            issue = ask_issues(q["id"])
            if issue:
                issues[q["id"]] = issue

    pass_count = sum(1 for v in answers.values() if v)
    pass_rate = pass_count / len(EVAL_QUESTIONS)

    print(f"\n{c('bold', 'RESULT')}: {pass_count}/{len(EVAL_QUESTIONS)} questions passed ({pass_rate:.0%})")
    if pass_rate >= 0.8:
        print(c("green", "  ✓ PASS (≥80%)"))
    else:
        print(c("red", "  ✗ FAIL (<80%)"))

    # Overall notes
    notes = input(f"\n{c('dim', 'Overall notes (optional, Enter to skip): ')}").strip()

    # Rewrite request
    rewrite_needed = not all(answers.values())
    if rewrite_needed:
        print(c("yellow", "\nMarked for rewrite based on failures."))

    return {
        "scenario_id": conversation.get("scenario_id"),
        "emotional_state": conversation.get("emotional_state"),
        "relationship_stage": conversation.get("relationship_stage"),
        "mode": conversation.get("mode"),
        "judge_total": conversation.get("judge", {}).get("total"),
        "answers": answers,
        "issues": issues,
        "notes": notes,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "passed": pass_rate >= 0.8,
        "rewrite_needed": rewrite_needed,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], reviewer: str):
    print("\n" + "═" * 70)
    print(c("bold", "HUMAN EVALUATION REPORT"))
    print(f"Reviewer: {reviewer}")
    print(f"Conversations evaluated: {len(results)}")

    if not results:
        print("No results.")
        return

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    overall_pass_rate = len(passed) / len(results)

    print(f"\nPassed (≥80%): {len(passed)}/{len(results)} ({overall_pass_rate:.1%})")
    print(f"Failed (<80%): {len(failed)}/{len(results)}")
    print(f"\nTarget: 80% pass rate | {'✓ MET' if overall_pass_rate >= 0.8 else '✗ NOT MET'}")

    # Per-question breakdown
    print(f"\n{c('bold', 'Per-question pass rates:')}")
    for q in EVAL_QUESTIONS:
        q_passed = sum(1 for r in results if r["answers"].get(q["id"], False))
        q_rate = q_passed / len(results)
        color = "green" if q_rate >= 0.8 else "red"
        print(f"  {c(color, f'{q_rate:.0%}')} — {q['question']}")

    # Issues found
    all_issues = {}
    for r in results:
        for q_id, issue in r.get("issues", {}).items():
            if issue:
                all_issues.setdefault(q_id, []).append(issue)

    if all_issues:
        print(f"\n{c('bold', 'Issues flagged:')}")
        for q_id, issue_list in all_issues.items():
            q_name = next(q["question"] for q in EVAL_QUESTIONS if q["id"] == q_id)
            print(f"\n  {q_name}:")
            for issue in issue_list:
                print(f"    - {issue}")

    print("─" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--output", default="human_eval_report.json")
    parser.add_argument("--reviewer", default="anonymous")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load and sample conversations
    all_convs = []
    with open(args.conversations) as f:
        for line in f:
            all_convs.append(json.loads(line.strip()))

    if args.seed:
        random.seed(args.seed)

    sample = random.sample(all_convs, min(args.sample, len(all_convs)))
    print(f"\nfumii Human Evaluation")
    print(f"Reviewer: {args.reviewer}")
    print(f"Conversations to review: {len(sample)}")
    print(f"\nInstructions:")
    print("  - Read each conversation carefully")
    print("  - Answer yes/no for each question")
    print("  - Be honest — the model learns from your feedback")
    print("  - If you flag an issue, describe it specifically")
    print("  - [Enter 's' at any question] to skip a conversation")
    print("  - [Ctrl+C] to quit and save progress\n")

    input(c("dim", "Press Enter to begin..."))

    results = []

    try:
        for i, conv in enumerate(sample):
            result = evaluate_conversation(conv, i + 1, len(sample))
            if result:
                results.append(result)

            print(f"\n{c('dim', 'Progress saved.')}")

            if i < len(sample) - 1:
                cont = input(c("dim", f"\nContinue to conversation {i+2}? [Enter / q to quit]: ")).strip().lower()
                if cont == "q":
                    break

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Saving {len(results)} results...")

    # Save results
    report = {
        "reviewer": args.reviewer,
        "evaluated_at": datetime.now().isoformat(),
        "conversations_evaluated": len(results),
        "overall_pass_rate": sum(1 for r in results if r["passed"]) / max(1, len(results)),
        "target_pass_rate": 0.80,
        "target_met": sum(1 for r in results if r["passed"]) / max(1, len(results)) >= 0.80,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print_report(results, args.reviewer)
    print(f"\nFull report saved to: {args.output}")

if __name__ == "__main__":
    main()
