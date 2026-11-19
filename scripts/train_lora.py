"""LoRA extractor training for BioEvidence."""

import json
import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from bioevidence.config import get_config
from bioevidence.scripts.create_sample_data import create_sample

logger = structlog.get_logger(__name__)


@dataclass
class LoRATrainConfig:
    """Configuration for LoRA extractor fine-tuning."""
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "models/qwen2.5-3b-lora-extractor"
    max_seq_length: int = 1024
    trust_remote_code: bool = True

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True

    # Data
    num_train_samples: int = 2700
    num_val_samples: int = 300
    num_test_samples: int = 200
    prompt_template: str = "chatml"
    max_input_length: int = 800
    max_output_length: int = 224

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.05
    fp16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2

    # Generation
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 256


def create_extraction_prompt(trial_text: str, schema: dict) -> str:
    """Create prompt for trial extraction."""
    schema_str = json.dumps(schema, indent=2)

    if True:  # chatml template
        return f"""<|im_start|>system
You are a clinical trial extraction assistant. Extract structured fields from the trial text and return valid JSON.
<|im_end|>
<|im_start|>user
Extract the following fields from this clinical trial text:

{trial_text}

Return JSON with these fields:
{schema_str}
<|im_end|>
<|im_start|>assistant
"""

    return f"""### Instruction:
Extract structured fields from the clinical trial text.

### Trial Text:
{trial_text}

### Schema:
{schema_str}

### Response:
"""


def prepare_extraction_data(config: LoRATrainConfig) -> tuple[Dataset, Dataset, Dataset]:
    """Prepare train/val/test datasets for extraction."""
    # For now, create synthetic data
    # In production, load from annotated data
    schema = {
        "primary_endpoint": "string",
        "endpoint_category": "string",
        "intervention_names": "array[string]",
        "intervention_class": "string",
        "randomized": "boolean",
        "blinding": "string",
        "population_age_group": "string",
        "key_inclusion": "array[string]",
        "key_exclusion": "array[string]",
    }

    # Create synthetic examples
    examples = []
    for i in range(config.num_train_samples + config.num_val_samples + config.num_test_samples):
        trial_text = f"Trial NCT{10000000 + i}: Phase 3 study of Drug X in Cancer. Primary endpoint: Overall Survival. Interventions: Drug X (monoclonal antibody). Randomized: Yes. Blinding: Double. Age: Adults. Inclusion: Age >= 18, Histologically confirmed cancer. Exclusion: Pregnancy, Active infection."
        prompt = create_extraction_prompt(trial_text, schema)
        target = json.dumps({
            "primary_endpoint": "Overall Survival",
            "endpoint_category": "overall_survival",
            "intervention_names": ["Drug X"],
            "intervention_class": "monoclonal_antibody",
            "randomized": True,
            "blinding": "double",
            "population_age_group": "adult",
            "key_inclusion": ["Age >= 18", "Histologically confirmed cancer"],
            "key_exclusion": ["Pregnancy", "Active infection"],
        })

        examples.append({"prompt": prompt, "target": target})

    # Split
    train_data = examples[:config.num_train_samples]
    val_data = examples[config.num_train_samples:config.num_train_samples + config.num_val_samples]
    test_data = examples[config.num_train_samples + config.num_val_samples:]

    def format_example(ex):
        return {"text": ex["prompt"] + ex["target"] + "<|im_end|>"}

    train_dataset = Dataset.from_list([format_example(e) for e in train_data])
    val_dataset = Dataset.from_list([format_example(e) for e in val_data])
    test_dataset = Dataset.from_list([format_example(e) for e in test_data])

    return train_dataset, val_dataset, test_dataset


def train_lora(config: LoRATrainConfig | None = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Train LoRA extractor."""
    config = config or LoRATrainConfig()

    if config.target_modules is None:
        config.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quantization config
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # For gradient checkpointing

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_extraction_data(config)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="mlflow" if True else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("lora_model_saved", path=str(output_dir))

    return model, tokenizer


def main():
    """Main entry point for LoRA training."""
    config = LoRATrainConfig()

    try:
        import mlflow
        mlflow.set_experiment("bioevidence-lora")
        mlflow.start_run()
        mlflow.log_params(config.__dict__)
    except ImportError:
        pass

    train_lora(config)

    try:
        import mlflow
        mlflow.end_run()
    except ImportError:
        pass


if __name__ == "__main__":
    main()