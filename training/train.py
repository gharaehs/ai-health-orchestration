"""
============================================================
Module 2 - QLoRA Fine-Tuning Script
AI Health Orchestration System
Model: meta-llama/Meta-Llama-3.1-8B-Instruct (fp16)
============================================================

What this script does:
  1. Loads Llama 3.1 8B fp16 in 4-bit via BitsAndBytes (QLoRA)
  2. Applies LoRA adapters to attention and MLP layers
  3. Fine-tunes on our health domain dataset
  4. Saves a lightweight adapter (~50-80MB) to models/adapters/health-v1/

Usage:
  source ~/.venv-training/bin/activate
  python training/train.py
"""

import json
import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. Load Config
# ============================================================

def load_config(config_path: str = "./training/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# 2. Load and Format Dataset
# ============================================================

# Llama 3.1 chat template format
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Input:
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

PROMPT_TEMPLATE_NO_INPUT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""


def format_example(example: dict) -> str:
    """Format a single training example into Llama 3.1 chat format."""
    instruction = example.get("instruction", "")
    input_data = example.get("input", {})
    output = example.get("output", {})

    input_str = json.dumps(input_data, indent=2) if input_data else ""
    output_str = json.dumps(output, indent=2) if isinstance(output, dict) else str(output)

    if input_str:
        return PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input_str,
            output=output_str
        )
    else:
        return PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output_str
        )


def load_dataset(dataset_path: str, train_split: float = 0.85):
    """Load JSONL dataset and split into train/eval."""
    logger.info(f"Loading dataset from {dataset_path}")

    examples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} examples")

    formatted = [{"text": format_example(ex)} for ex in examples]

    split_idx = int(len(formatted) * train_split)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    logger.info(f"Train: {len(train_data)} examples | Eval: {len(eval_data)} examples")
    logger.info("--- Sample training example (first 500 chars) ---")
    logger.info(train_data[0]["text"][:500])
    logger.info("---")

    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


# ============================================================
# 3. Load Model in 4-bit (QLoRA)
# ============================================================

def load_model_and_tokenizer(config: dict):
    """Load Llama 3.1 8B fp16 in 4-bit quantization via BitsAndBytes."""
    model_path = config["model"].get("local_path", config["model"]["name"])
    quant_cfg = config["quantization"]

    logger.info(f"Loading model from: {model_path}")
    logger.info("Quantization: 4-bit NF4 (QLoRA via BitsAndBytes)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer


# ============================================================
# 4. Apply LoRA
# ============================================================

def apply_lora(model, config: dict):
    """Prepare model for QLoRA training and apply LoRA adapters."""
    lora_cfg = config["lora"]

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    logger.info("LoRA adapters applied successfully")

    return model


# ============================================================
# 5. Train
# ============================================================

def train(config: dict, model, tokenizer, train_dataset, eval_dataset):
    """Run the fine-tuning."""
    train_cfg = config["training"]

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg["fp16"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["dataset"]["max_seq_length"],
        packing=False,
    )

    logger.info("Starting training...")
    logger.info(f"Output will be saved to: {train_cfg['output_dir']}")

    trainer.train()

    logger.info("Training complete. Saving adapter...")
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    logger.info(f"Adapter saved to {train_cfg['output_dir']}")
    return trainer


# ============================================================
# 6. Quick Inference Test
# ============================================================

def test_inference(model, tokenizer, test_prompt: str):
    """Run a quick test generation after training."""
    model.eval()

    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{test_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the input prompt
    response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    return response


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AI Health Orchestration - Module 2: QLoRA Fine-Tuning")
    logger.info("Model: meta-llama/Meta-Llama-3.1-8B-Instruct")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. This script requires a CUDA GPU.")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = load_config("./training/config.yaml")

    train_dataset, eval_dataset = load_dataset(
        config["dataset"]["path"],
        config["dataset"]["train_split"]
    )

    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)
    trainer = train(config, model, tokenizer, train_dataset, eval_dataset)

    logger.info("\n--- Post-training inference test ---")
    test_prompt = "Generate a meal plan for a 30-year-old male, 80kg, goal: fat loss, 2000 kcal/day"
    response = test_inference(model, tokenizer, test_prompt)
    logger.info(f"Test response:\n{response}")

    logger.info("\nModule 2 complete!")
    logger.info(f"Your LoRA adapter is saved at: {config['training']['output_dir']}")
    logger.info("Next: Run evaluate.py to compare base model vs fine-tuned model")