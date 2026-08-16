"""
fumii SFT Training — Unsloth (Phase 1)
Faster alternative to Axolotl for single-GPU setups.

Requires: pip install unsloth transformers datasets trl

Run:
    python train_unsloth.py \
        --dataset sft_chatml.jsonl \
        --output ./checkpoints/fumii-sft-v1 \
        --epochs 3

On a single A100 (80GB): ~2–3 hours for 1,400 conversations.
On a 4090 (24GB): ~4–5 hours with gradient checkpointing.
"""

import json
import sys
import argparse
from pathlib import Path
from datasets import Dataset

from fumii_pipeline.utils.constants import FUMII_SYSTEM_PROMPT  # noqa: E402

def load_chatml_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            convs = obj.get("conversations", [])
            if not convs:
                continue

            # Format as a single text string in ChatML format
            text = ""
            for msg in convs:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

            records.append({"text": text, "metadata": obj.get("metadata", {})})

    return Dataset.from_list(records)

def train(dataset_path: str, output_dir: str, num_epochs: int, max_seq_length: int):
    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        raise ImportError(
            "Unsloth not installed. Run: pip install unsloth transformers datasets trl"
        )

    print("Loading base model: Qwen/Qwen2.5-3B-Instruct")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,           # Auto-detect BF16/FP16
        load_in_4bit=True,    # QLoRA — fits on 16GB GPU
    )

    print("Adding LoRA adapters...")
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
        use_rslora=False,
    )

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_chatml_dataset(dataset_path)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="wandb",
        run_name="fumii-sft-v1",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("\nStarting SFT training...")
    print(f"  Base model:  Qwen/Qwen2.5-3B-Instruct")
    print(f"  LoRA rank:   16")
    print(f"  Epochs:      {num_epochs}")
    print(f"  Train size:  {len(train_dataset)}")
    print(f"  Output:      {output_dir}")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining complete. Model saved to {output_dir}")
    print("Next step: python export_gguf.py --checkpoint " + output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sft_chatml.jsonl")
    parser.add_argument("--output", default="./checkpoints/fumii-sft-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    train(args.dataset, args.output, args.epochs, args.max_seq_length)

if __name__ == "__main__":
    main()
