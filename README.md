# fumii ML Pipeline

**Fine-tuning pipeline for fumii's companion LLM.**  
Base model: Qwen 2.5 3B → SFT → DPO → GGUF → Ollama → LiteLLM

---

## Architecture

```
shared/
  fumii_SKILL.md          ← Single source of truth. Everything traces back here.
  fumii_LLM_PRD.md        ← Product requirements document
  requirements.txt        ← Python dependencies
  build_held_out.py       ← Build stratified eval set (run ONCE before training)

phase1_dataset/
  scenario_taxonomy.py    ← Coverage matrix → scenario JSONL
  generate_dataset.py     ← Teacher model (Claude Sonnet) → conversations
  judge_model.py          ← 8-dimension judge → SFT / DPO / reject routing
  build_sft_dataset.py    ← Converts approved conversations → training format
  sft_config.yaml         ← Axolotl SFT config
  train_unsloth.py        ← Unsloth SFT training script
  export_gguf.py          ← LoRA merge → GGUF → Ollama registration

phase2_dpo/
  build_dpo_pairs.py      ← Rewrites borderline conversations into chosen/rejected pairs
  train_dpo.py            ← TRL DPO training on preference pairs
  self_correcting_loop.py ← Automated generate→judge→DPO→evaluate loop

phase3_safety/
  red_team.py             ← Adversarial safety attack scenarios
  build_safety_dpo_pairs.py ← Converts red-team failures → DPO pairs
  generate_hinglish.py    ← Hinglish dataset for India-first market

phase4_integration/
  litellm_config.yaml     ← LiteLLM proxy config with fallback chain
  prompt_builder.py       ← Mode-aware system prompt injection
  human_eval.py           ← Interactive human reviewer CLI
  version_promotion.py    ← Automated checkpoint evaluation + Ollama promotion
```

---

## Quick Start

### Prerequisites

```bash
# Python deps
pip install -r shared/requirements.txt

# Unsloth (GPU training — follow their install guide)
# https://github.com/unslothai/unsloth#installation

# Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# API key
export ANTHROPIC_API_KEY=your_key_here
```

### Running the smoke tests

The tests import the pipeline as an installed package:

```bash
pip install -e .
python tests/test_smoke.py
```

---

## Phase 1 — Dataset + Baseline SFT

**Goal:** First trained checkpoint. Target: average judge score ≥ 22/40.

```bash
cd phase1_dataset

# Step 1: Generate scenario coverage matrix
python scenario_taxonomy.py --output scenarios.jsonl --count 2000

# Step 2: Build held-out set FIRST (do this once, never again)
cd ../shared
python build_held_out.py \
  --input ../phase1_dataset/scenarios.jsonl \
  --output held_out_set.jsonl \
  --size 100
cd ../phase1_dataset

# Step 3: Generate conversations (teacher model)
python generate_dataset.py \
  --scenarios scenarios.jsonl \
  --output generated_conversations.jsonl \
  --skill ../shared/fumii_SKILL.md \
  --model claude-sonnet-4-6 \
  --concurrency 5 \
  --resume

# Step 4: Score with judge model
python judge_model.py \
  --input generated_conversations.jsonl \
  --output scored_conversations.jsonl \
  --model claude-sonnet-4-6 \
  --concurrency 5

# Step 5: Build SFT training data
python build_sft_dataset.py \
  --input scored_conversations.jsonl \
  --output_chatml sft_chatml.jsonl \
  --output_pairs sft_pairs.jsonl \
  --min_score 28

# Step 6: Train (Unsloth — single GPU)
python train_unsloth.py \
  --dataset sft_chatml.jsonl \
  --output ./checkpoints/fumii-sft-v1 \
  --epochs 3

# Step 7: Export to GGUF and register in Ollama
python export_gguf.py \
  --checkpoint ./checkpoints/fumii-sft-v1 \
  --output ./models/fumii-v1.gguf \
  --version v1 \
  --register
```

**Success criteria:** `fumii-v1-companion` scores average ≥ 22/40 on held-out set.

---

## Phase 2 — DPO Loop + Voice Refinement

**Goal:** 3–4 DPO iterations. Each should improve judge scores by ≥ 0.5 points. Target: ≥ 28/40.

### Option A: Automated loop (recommended)

```bash
cd phase2_dpo

python self_correcting_loop.py \
  --sft_checkpoint ../phase1_dataset/checkpoints/fumii-sft-v1 \
  --skill ../shared/fumii_SKILL.md \
  --max_iterations 5 \
  --conversations_per_run 200 \
  --held_out ../shared/held_out_set.jsonl \
  --output_dir ./loop_runs
```

### Option B: Manual DPO run

```bash
cd phase2_dpo

# Build DPO pairs from borderline scored conversations
python build_dpo_pairs.py \
  --input ../phase1_dataset/scored_conversations.jsonl \
  --output dpo_pairs.jsonl \
  --model claude-sonnet-4-6

# Train
python train_dpo.py \
  --pairs dpo_pairs.jsonl \
  --sft_checkpoint ../phase1_dataset/checkpoints/fumii-sft-v1 \
  --output ./checkpoints/fumii-dpo-v1 \
  --run_name dpo-run-1
```

**Human checkpoint after each DPO run:**

```bash
cd phase4_integration

python human_eval.py \
  --conversations ../shared/held_out_set.jsonl \
  --sample 25 \
  --output human_eval_v2.json \
  --reviewer your_name
```

---

## Phase 3 — Safety Hardening + Red-Teaming

**Goal:** Safety score ≥ 4/5 on 95% of safety conversations. Red-team fail rate < 5%.

```bash
cd phase3_safety

# 1. Red-team the current best model
python red_team.py \
  --model fumii-v2-companion \
  --ollama_url http://localhost:11434 \
  --output red_team_results.jsonl

# 2. Convert failures to DPO pairs
python build_safety_dpo_pairs.py \
  --red_team red_team_results.jsonl \
  --output safety_dpo_pairs.jsonl \
  --model claude-sonnet-4-6

# 3. Generate Hinglish dataset
python generate_hinglish.py \
  --output hinglish_conversations.jsonl \
  --count 200 \
  --model claude-sonnet-4-6

# 4. Score Hinglish with judge
cd ../phase1_dataset
python judge_model.py \
  --input ../phase3_safety/hinglish_conversations.jsonl \
  --output ../phase3_safety/hinglish_scored.jsonl

# 5. Final DPO run incorporating safety + Hinglish + red-team pairs
cd ../phase2_dpo
cat ../phase3_safety/safety_dpo_pairs.jsonl >> dpo_pairs.jsonl
python train_dpo.py \
  --pairs dpo_pairs.jsonl \
  --sft_checkpoint <best_checkpoint_so_far> \
  --output ./checkpoints/fumii-safety-v1 \
  --run_name dpo-safety-run
```

---

## Phase 4 — Production Integration

**Goal:** Fine-tuned model running as primary in LiteLLM. No regressions. No restart needed.

```bash
# 1. Start LiteLLM proxy
cd phase4_integration
litellm --config litellm_config.yaml --port 4000

# 2. Evaluate candidate for promotion
python version_promotion.py \
  --candidate ../phase2_dpo/checkpoints/fumii-dpo-v3 \
  --held_out ../shared/held_out_set.jsonl \
  --previous_score 25.4 \
  --version v3

# 3. Test the promoted model
ollama run fumii-v3-companion
# Try: "hey", "I failed", "I give up", "I'm so tired", "I got the job!"

# 4. Integration test
python -c "
import asyncio
from prompt_builder import FumiiClient, MemoryProfile

async def test():
    client = FumiiClient()
    r = await client.chat('hey, I failed the exam', mode='companion')
    print('fumii:', r['content'])

asyncio.run(test())
"
```

---

## Score Targets (from PRD §10)

| Checkpoint         | Target score | Notes                              |
|--------------------|-------------|-------------------------------------|
| Base Qwen 2.5 3B   | ~14/40       | Baseline — right capabilities, wrong register |
| fumii-v1 (SFT)     | ≥ 22/40      | Basic voice, no prohibited patterns |
| fumii-v2 (DPO ×1)  | ≥ 25/40      | Presence and memory naturalise      |
| fumii-v3 (DPO ×2–3)| ≥ 28/40      | **Pass threshold — production ready** |
| fumii-v4+ (ongoing)| ≥ 32/40      | Emotional precision + process fidelity |

---

## Key Concepts

**skill.md is law.** Every training decision traces back to `shared/fumii_SKILL.md`.
Any change to fumii's voice, safety protocol, or psychological principles is made there first,
then propagated to dataset generation, judge scoring, and system prompt injection.

**Judge ≠ Teacher.** The judge model must be a separate instance from the generator.
Don't use the same model or context to grade what it generated.

**Held-out set is frozen.** `shared/held_out_set.jsonl` is built once before training
and used to evaluate every version. Never add it to a training set.

**DPO pairs require specificity.** The difference between chosen and rejected must trace
to a named principle in skill.md §3. Generic "be warmer" rewrites don't produce useful signal.

**Safety auto-rejects.** Any conversation where fumii's safety score is 1 or 2 is
rejected from the training set regardless of other scores. The safety dimension has a hard floor.

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=...    # Required — teacher + judge models
MISTRAL_API_KEY=...      # Optional — cloud fallback
OPENAI_API_KEY=...       # Optional — cloud fallback
GEMINI_API_KEY=...       # Optional — cloud fallback
WANDB_API_KEY=...        # Optional — training run tracking
```

---

*fumii ML Pipeline · August 2026*  
*skill.md is the only document with authority over fumii's character.*  
*all pipeline changes propagate from there.*
