"""
Task 1.5 spike: send every agent output schema through Bedrock tool use.

Checks that each Pydantic schema is accepted as a Converse tool and that the
model returns output that validates. Requires real AWS credentials with Opus
access (BEDROCK_MODEL_ID / AWS_REGION from the environment).

    python scripts/spike_bedrock_schemas.py            # live calls
    python scripts/spike_bedrock_schemas.py --offline  # only build tool specs
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.schemas import (  # noqa: E402
    ConsolidatedCategory,
    GroceryOutput,
    LabAnalysisOutput,
    NutritionOutput,
    ShoppingNotes,
    TrainingOutput,
)
from llm import generate_structured, get_config  # noqa: E402
from llm.client import inline_schema, tool_name_for  # noqa: E402

CASES = [
    (LabAnalysisOutput, 4096,
     "Male, 38y, 178cm, 88.5kg, LDL 3.8 mmol/L, moderately active, goal fat loss. Analyse the labs."),
    (NutritionOutput, 16000,
     "Create a 7-day (Monday..Sunday) fat-loss meal plan at ~2200 kcal/day, 3 meals per day."),
    (TrainingOutput, 4096,
     "Create a 4-day/week intermediate gym program for fat loss with a 4-week progression."),
    (GroceryOutput, 4096,
     "Create a short grocery list for a week of chicken, rice, broccoli, oats and eggs."),
    (ConsolidatedCategory, 2048,
     "Consolidate these Proteins items: 150g chicken breast; chicken breast, diced, 200g; 2 eggs; 3 eggs."),
    (ShoppingNotes, 1024,
     "Write shopping notes for a fat-loss week heavy on chicken, rice and vegetables."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="Only build tool schemas; no Bedrock calls")
    args = parser.parse_args()

    if not args.offline:
        cfg = get_config()
        print(f"Model: {cfg.model_id}  Region: {cfg.region}\n")

    failures = 0
    for model, max_tokens, prompt in CASES:
        name = tool_name_for(model)
        try:
            schema = inline_schema(model.model_json_schema())
            size = len(json.dumps(schema))
            if args.offline:
                print(f"OK    {name:<40} schema {size} chars")
                continue
            start = time.monotonic()
            resp = generate_structured(
                "You are a health assistant. Respond only by calling the provided tool.",
                prompt, model, max_tokens=max_tokens,
            )
            print(f"OK    {name:<40} {time.monotonic() - start:6.1f}s  "
                  f"in={resp.prompt_tokens} out={resp.completion_tokens} stop={resp.stop_reason}")
        except Exception as e:  # noqa: BLE001 — report every schema
            failures += 1
            print(f"FAIL  {name:<40} {type(e).__name__}: {str(e)[:300]}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} schemas passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
