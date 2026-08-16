import os
import re
from pathlib import Path

ROOT = Path("d:/fumii-pipeline")

REPLACEMENTS = {
    # Utils
    r"from constants import ": "from fumii_pipeline.utils.constants import ",
    r"from pipeline_utils import ": "from fumii_pipeline.utils.pipeline_utils import ",
    r"import pipeline_utils": "from fumii_pipeline.utils import pipeline_utils",
    
    # Data
    r"from scenario_taxonomy import ": "from fumii_pipeline.data.scenario_taxonomy import ",
    r"from build_sft_dataset import ": "from fumii_pipeline.data.build_sft_dataset import ",
    r"from build_held_out import ": "from fumii_pipeline.data.build_held_out import ",
    r"from generate_dataset import ": "from fumii_pipeline.data.generate_dataset import ",
    
    # Training
    r"from build_dpo_pairs import ": "from fumii_pipeline.training.build_dpo_pairs import ",
    
    # Safety
    r"from red_team import ": "from fumii_pipeline.safety.red_team import ",
    r"from build_safety_dpo_pairs import ": "from fumii_pipeline.safety.build_safety_dpo_pairs import ",
    r"from generate_hinglish import ": "from fumii_pipeline.safety.generate_hinglish import ",
    
    # Evaluation
    r"from judge_model import ": "from fumii_pipeline.evaluation.judge_model import ",
    
    # Deployment
    r"from prompt_builder import ": "from fumii_pipeline.deployment.prompt_builder import ",
    r"from export_gguf import ": "from fumii_pipeline.deployment.export_gguf import "
}

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove sys.path.insert lines
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if "sys.path.insert" in line:
            continue
        new_lines.append(line)
    
    content = "\n".join(new_lines)

    # Apply all module replacements
    for old, new in REPLACEMENTS.items():
        content = re.sub(old, new, content)

    # Clean up empty lines where sys.path.insert used to be
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    dirs_to_check = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
    for d in dirs_to_check:
        for filepath in d.rglob("*.py"):
            print(f"Processing {filepath}")
            process_file(filepath)

if __name__ == "__main__":
    main()
