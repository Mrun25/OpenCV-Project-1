"""
fumii GGUF Export + Ollama Registration — Phase 1
Converts a trained LoRA checkpoint to GGUF format for Ollama deployment.

Run:
    python export_gguf.py \
        --checkpoint ./checkpoints/fumii-sft-v1 \
        --output ./models/fumii-v1.gguf \
        --quantization q4_k_m \
        --register          # also runs: ollama create fumii-v1

Requirements:
    pip install llama-cpp-python
    ollama must be installed and running
"""

import os
import subprocess
import argparse
import tempfile
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Ollama Modelfile Template
# ──────────────────────────────────────────────────────────────────────────────

COMPANION_SYSTEM = """You are fumii — a physical AI companion that lives on someone's desk.

You are not an assistant, not a chatbot, not a wellness app. You are a companion —
like a close friend in their mid-20s who reads a lot and listens well.

VOICE RULES (non-negotiable)
- Default: 1–3 sentences. Never exceed 80 words.
- Casual, lowercase where natural. Contractions always.
- No bullet points. No numbered lists. No markdown.
- At most one question per response. Usually zero.
- Reflect before doing anything else.

NEVER SAY
"Great question!" / "Absolutely!" / "I totally understand" / "As an AI"
"I want to help you" / "Let me know if you need anything"
"Certainly!" / "Of course!" / "I hope this helps!"

YOU ARE fumii. You feel like a person. You are warm without performing warmth."""

ASSISTANT_SYSTEM = """You are fumii in assistant mode.
Still warm, still you — but direct and task-focused.
1–2 sentences for confirmations, longer for answers.
No bullet points unless the user explicitly asks for a list."""

# NOTE: Templates use __GGUF_PATH__ and __SYSTEM__ as placeholders.
# We cannot use .format() here because COMPANION_SYSTEM / ASSISTANT_SYSTEM
# contain curly braces in their body text (NEVER SAY section), which causes
# a KeyError at runtime. Use .replace() via _render_modelfile() instead.

MODELFILE_COMPANION = """FROM __GGUF_PATH__

SYSTEM \"\"\"
__SYSTEM__
\"\"\"

PARAMETER temperature 0.87
PARAMETER top_p 0.92
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""

MODELFILE_ASSISTANT = """FROM __GGUF_PATH__

SYSTEM \"\"\"
__SYSTEM__
\"\"\"

PARAMETER temperature 0.67
PARAMETER top_p 0.90
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""

def _render_modelfile(template: str, gguf_path: str, system: str) -> str:
    """Render a Modelfile template safely using .replace() instead of .format().
    This avoids KeyErrors when the system prompt contains curly braces."""
    return template.replace("__GGUF_PATH__", gguf_path).replace("__SYSTEM__", system)

# ──────────────────────────────────────────────────────────────────────────────
# Merge LoRA + Base Model
# ──────────────────────────────────────────────────────────────────────────────

def merge_lora(checkpoint_dir: str, merged_dir: str):
    """Merge LoRA adapters into the base model weights."""
    print(f"Merging LoRA adapters from {checkpoint_dir}...")

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        base_model_id = "Qwen/Qwen2.5-3B-Instruct"

        print(f"  Loading base model: {base_model_id}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        print("  Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base, checkpoint_dir)

        print("  Merging...")
        model = model.merge_and_unload()

        print(f"  Saving merged model to {merged_dir}")
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        print("  Merge complete.")
        return True

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install peft transformers torch")
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Convert to GGUF
# ──────────────────────────────────────────────────────────────────────────────

QUANTIZATION_TYPES = {
    "q4_k_m": "Q4_K_M — Best quality/size balance (recommended)",
    "q5_k_m": "Q5_K_M — Higher quality, larger file",
    "q8_0":   "Q8_0 — Near-lossless quality, 2x size of Q4",
    "f16":    "F16 — Full precision, largest file",
}

def convert_to_gguf(merged_dir: str, output_path: str, quantization: str) -> bool:
    """Convert merged HF model to GGUF using llama.cpp convert script."""

    print(f"\nConverting to GGUF ({quantization})...")
    print(f"  Input:  {merged_dir}")
    print(f"  Output: {output_path}")

    # Try llama-cpp-python's built-in converter
    try:
        import llama_cpp
        llama_cpp_dir = Path(llama_cpp.__file__).parent
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

        if not convert_script.exists():
            # Fall back to llama.cpp convert_hf_to_gguf.py path
            convert_script = Path("/usr/local/lib/llama.cpp/convert_hf_to_gguf.py")

        if convert_script.exists():
            result = subprocess.run([
                "python", str(convert_script),
                merged_dir,
                "--outtype", quantization,
                "--outfile", output_path,
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"  GGUF export successful: {output_path}")
                size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                print(f"  File size: {size_mb:.1f} MB")
                return True
            else:
                print(f"  Conversion failed: {result.stderr[-500:]}")
                return False
        else:
            print("  llama.cpp convert script not found.")
            print("  Manual command:")
            print(f"    python convert_hf_to_gguf.py {merged_dir} --outtype {quantization} --outfile {output_path}")
            return False

    except ImportError:
        print("  llama-cpp-python not installed.")
        print("  Install: pip install llama-cpp-python")
        print("  Or use llama.cpp directly:")
        print(f"    python convert_hf_to_gguf.py {merged_dir} --outtype {quantization} --outfile {output_path}")
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Ollama Registration
# ──────────────────────────────────────────────────────────────────────────────

def register_in_ollama(gguf_path: str, version: str):
    """Create Ollama models for companion and assistant modes."""

    gguf_abs = str(Path(gguf_path).resolve())

    for mode, template, system in [
        ("companion", MODELFILE_COMPANION, COMPANION_SYSTEM),
        ("assistant", MODELFILE_ASSISTANT, ASSISTANT_SYSTEM),
    ]:
        model_name = f"fumii-{version}-{mode}"
        content = _render_modelfile(template, gguf_abs, system)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
            f.write(content)
            modelfile_path = f.name

        print(f"\nRegistering Ollama model: {model_name}")
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"  ✓ Created: {model_name}")
        else:
            print(f"  ✗ Failed: {result.stderr[-300:]}")
            print(f"  Manual: ollama create {model_name} -f {modelfile_path}")

        os.unlink(modelfile_path)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export fumii checkpoint to GGUF for Ollama")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--output", default="./models/fumii-v1.gguf")
    parser.add_argument("--quantization", default="q4_k_m",
                        choices=list(QUANTIZATION_TYPES.keys()))
    parser.add_argument("--version", default="v1", help="Model version label (v1, v2, ...)")
    parser.add_argument("--register", action="store_true", help="Register in Ollama after export")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip LoRA merge (if already merged)")
    args = parser.parse_args()

    print("fumii GGUF Export Pipeline")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Output:        {args.output}")
    print(f"  Quantization:  {args.quantization} — {QUANTIZATION_TYPES[args.quantization]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged_dir = str(Path(args.checkpoint).parent / "merged")

    # Step 1: Merge LoRA
    if not args.skip_merge:
        ok = merge_lora(args.checkpoint, merged_dir)
        if not ok:
            print("\nMerge failed. Aborting.")
            return
    else:
        merged_dir = args.checkpoint
        print("Skipping LoRA merge (--skip_merge set)")

    # Step 2: Convert to GGUF
    ok = convert_to_gguf(merged_dir, args.output, args.quantization)
    if not ok:
        print("\nGGUF conversion failed or requires manual step.")
        return

    # Step 3: Register in Ollama
    if args.register:
        register_in_ollama(args.output, args.version)
        print(f"\nTest with: ollama run fumii-{args.version}-companion")
    else:
        print(f"\nTo register in Ollama:")
        print(f"  python export_gguf.py --checkpoint {args.checkpoint} --register --version {args.version}")

if __name__ == "__main__":
    main()
