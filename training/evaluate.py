"""
============================================================
Module 2 - Evaluation: Base Model vs Fine-Tuned
AI Health Orchestration System
Model: meta-llama/Meta-Llama-3.1-8B-Instruct
============================================================

Run this AFTER training to compare outputs side by side.

Usage:
  source ~/.venv-training/bin/activate
  python training/evaluate.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "./models/llama"
ADAPTER_PATH = "./models/adapters/health-v1"

TEST_PROMPTS = [
    {
        "name": "Meal Plan - Fat Loss Male",
        "prompt": "Generate a meal plan for a 35-year-old male, 90kg, goal: fat loss, 2000 kcal/day, no dietary restrictions. Return as JSON."
    },
    {
        "name": "Meal Plan - Vegetarian Female",
        "prompt": "Generate a meal plan for a 27-year-old female, 62kg, goal: muscle gain, 2300 kcal/day, vegetarian. Return as JSON."
    },
    {
        "name": "Gym Program - Beginner",
        "prompt": "Generate a gym program for a 25-year-old female, beginner, goal: fat loss, 3 days per week, home dumbbells only. Return as JSON with sets, reps, rest."
    },
    {
        "name": "Grocery List",
        "prompt": "Generate a grocery list for a male, fat loss goal, 2000 kcal/day, one week, no dietary restrictions. Return as JSON organised by category."
    },
    {
        "name": "Blood Test Analysis",
        "prompt": "Interpret these blood test results for a 45-year-old male: LDL 4.8 mmol/L, fasting glucose 6.4 mmol/L, vitamin D 32 nmol/L, ferritin 45 ug/L, HbA1c 6.1%. Return dietary and exercise constraints as JSON."
    },
]


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model():
    logger.info("Loading BASE model (no fine-tuning)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_finetuned_model():
    logger.info("Loading FINE-TUNED model (base + LoRA adapter)...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 600) -> str:
    formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()


def is_valid_json(text: str) -> bool:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1:
            return False
        json.loads(text[start:end])
        return True
    except Exception:
        return False


def evaluate():
    # --- Base model ---
    base_model, tokenizer = load_base_model()
    base_model.eval()

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING BASE MODEL")
    logger.info("=" * 60)

    base_results = {}
    for test in TEST_PROMPTS:
        logger.info(f"\nTest: {test['name']}")
        response = generate(base_model, tokenizer, test["prompt"])
        valid_json = is_valid_json(response)
        base_results[test["name"]] = {
            "response": response,
            "valid_json": valid_json,
            "length": len(response)
        }
        logger.info(f"  Valid JSON: {valid_json}")
        logger.info(f"  Response length: {len(response)} chars")
        logger.info(f"  Preview: {response[:200]}...")

    del base_model
    torch.cuda.empty_cache()
    logger.info("\nBase model unloaded. GPU memory cleared.")

    # --- Fine-tuned model ---
    ft_model, tokenizer = load_finetuned_model()
    ft_model.eval()

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING FINE-TUNED MODEL (LoRA adapter)")
    logger.info("=" * 60)

    ft_results = {}
    for test in TEST_PROMPTS:
        logger.info(f"\nTest: {test['name']}")
        response = generate(ft_model, tokenizer, test["prompt"])
        valid_json = is_valid_json(response)
        ft_results[test["name"]] = {
            "response": response,
            "valid_json": valid_json,
            "length": len(response)
        }
        logger.info(f"  Valid JSON: {valid_json}")
        logger.info(f"  Response length: {len(response)} chars")
        logger.info(f"  Preview: {response[:200]}...")

    # --- Comparison Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Test':<35} {'Base JSON':>10} {'FT JSON':>10} {'Improvement':>12}")
    logger.info("-" * 70)

    json_improvements = 0
    for test in TEST_PROMPTS:
        name = test["name"]
        base_json = base_results[name]["valid_json"]
        ft_json = ft_results[name]["valid_json"]
        improved = "✅ YES" if (ft_json and not base_json) else ("same" if base_json == ft_json else "⚠️ regressed")
        if ft_json and not base_json:
            json_improvements += 1
        logger.info(f"{name:<35} {str(base_json):>10} {str(ft_json):>10} {improved:>12}")

    logger.info("-" * 70)
    logger.info(f"JSON consistency improvements: {json_improvements}/{len(TEST_PROMPTS)}")

    output = {
        "base_model": base_results,
        "finetuned_model": ft_results,
        "summary": {
            "total_tests": len(TEST_PROMPTS),
            "base_valid_json": sum(1 for r in base_results.values() if r["valid_json"]),
            "ft_valid_json": sum(1 for r in ft_results.values() if r["valid_json"]),
        }
    }

    with open("./training/evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("\nFull results saved to: ./training/evaluation_results.json")
    logger.info("Module 2 evaluation complete!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    evaluate()