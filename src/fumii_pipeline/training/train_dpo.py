"""
fumii DPO Training — Phase 2
Direct Preference Optimisation on preference pairs.

Run after SFT (Phase 1). Requires fumii-sft-v1 checkpoint.

Run:
    python train_dpo.py \
        --pairs dpo_pairs.jsonl \
        --sft_checkpoint ../phase1_dataset/checkpoints/fumii-sft-v1 \
        --output ./checkpoints/fumii-dpo-v1 \
        --run_name dpo-run-1

Requires: pip install trl peft transformers unsloth
"""

import json
import sys
import argparse
from pathlib import Path
from datasets import Dataset

from fumii_pipeline.utils.constants import FUMII_SYSTEM_PROMPT  # noqa: E402  # imported for re-export consistency

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────────────────────────────────────

def chatml_to_string(messages: list[dict]) -> str:
    """Convert ChatML message list to a single string for DPO training."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text

def load_dpo_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())

            chosen = obj.get("chosen", [])
            rejected = obj.get("rejected", [])

            if not chosen or not rejected:
                continue

            records.append({
                "prompt": chatml_to_string(chosen[:-1]),    # Everything up to last assistant turn
                "chosen": chosen[-1]["content"],             # Last assistant turn (chosen)
                "rejected": rejected[-1]["content"],         # Last assistant turn (rejected)
                "metadata": {
                    "scenario_id": obj.get("scenario_id"),
                    "emotional_state": obj.get("emotional_state"),
                    "relationship_stage": obj.get("relationship_stage"),
                    "mode": obj.get("mode"),
                    "what_changed": obj.get("what_changed"),
                    "improvement_principle": obj.get("improvement_principle"),
                }
            })

    return Dataset.from_list(records)

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    pairs_path: str,
    sft_checkpoint: str,
    output_dir: str,
    run_name: str,
    beta: float,
    max_length: int,
):
    try:
        from unsloth import FastLanguageModel, PatchDPOTrainer
        from trl import DPOTrainer, DPOConfig
        import torch
        PatchDPOTrainer()  # Unsloth's DPO patch for speed
    except ImportError as exc:
        raise ImportError(
            "Required: pip install 'unsloth>=2024.9' 'trl>=0.8' transformers peft torch\n"
            f"Original error: {exc}"
        ) from exc

    print(f"Loading SFT checkpoint: {sft_checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_checkpoint,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Enable LoRA for DPO — same config as SFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"Loading DPO pairs from {pairs_path}...")
    dataset = load_dpo_dataset(pairs_path)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train pairs: {len(train_dataset)} | Eval pairs: {len(eval_dataset)}")

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,           # 1 epoch per DPO run is usually enough
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,           # Very low — DPO is a subtle adjustment
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=5,
        report_to="wandb",
        run_name=run_name,
        beta=beta,                    # KL penalty — keeps model close to SFT
        loss_type="sigmoid",          # Standard DPO loss
        max_length=max_length,
        max_prompt_length=max_length // 2,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,               # Unsloth handles reference model internally
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print(f"\nStarting DPO training...")
    print(f"  SFT base:   {sft_checkpoint}")
    print(f"  Beta (KL):  {beta}")
    print(f"  Pairs:      {len(train_dataset)}")
    print(f"  Output:     {output_dir}")
    print(f"  Run name:   {run_name}")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nDPO training complete. Checkpoint saved to {output_dir}")
    print(f"Next: python ../phase1_dataset/export_gguf.py --checkpoint {output_dir} --version {run_name} --register")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="dpo_pairs.jsonl")
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--output", default="./checkpoints/fumii-dpo-v1")
    parser.add_argument("--run_name", default="dpo-run-1")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty — 0.1 is standard. Increase to stay closer to SFT.")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    train(args.pairs, args.sft_checkpoint, args.output, args.run_name, args.beta, args.max_length)

if __name__ == "__main__":
    main()
