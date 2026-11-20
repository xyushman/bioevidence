"""Constrained JSON decoding for extraction using Outlines (optional)."""

import json
from typing import Any

import structlog

# Try to import Outlines (optional dependency)
try:
    import outlines
    from outlines import models as outlines_models
    from outlines.samplers import greedy

    OUTLINES_AVAILABLE = True
except ImportError:
    outlines = None
    outlines_models = None
    greedy = None
    OUTLINES_AVAILABLE = False

from bioevidence.extraction.prepare_data import (
    AGE_GROUPS,
    BLINDING_TYPES,
    ENDPOINT_CATEGORIES,
    INTERVENTION_CLASSES,
    create_extraction_prompt,
)

logger = structlog.get_logger(__name__)


# Define JSON schema for constrained decoding
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_endpoint": {"type": "string"},
        "endpoint_category": {"type": "string", "enum": ENDPOINT_CATEGORIES},
        "intervention_names": {"type": "array", "items": {"type": "string"}},
        "intervention_class": {"type": "string", "enum": INTERVENTION_CLASSES},
        "randomized": {"type": "boolean"},
        "blinding": {"type": "string", "enum": BLINDING_TYPES},
        "population_age_group": {"type": "string", "enum": AGE_GROUPS},
        "key_inclusion": {"type": "array", "items": {"type": "string"}},
        "key_exclusion": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "primary_endpoint",
        "endpoint_category",
        "intervention_names",
        "intervention_class",
        "randomized",
        "blinding",
        "population_age_group",
        "key_inclusion",
        "key_exclusion",
    ],
    "additionalProperties": False,
}


class ConstrainedExtractor:
    """Extractor with constrained JSON decoding using Outlines."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "auto"):
        if not OUTLINES_AVAILABLE:
            raise RuntimeError("Outlines not installed. Run: uv add outlines")

        self.model_name = model_name
        self.device = device
        self._model = None
        self._generator = None

    @property
    def model(self):
        """Lazy load Outlines model."""
        if self._model is None:
            self._model = outlines_models.transformers(
                self.model_name,
                device=self.device,
                model_kwargs={
                    "torch_dtype": "auto",
                    "trust_remote_code": True,
                },
            )
        return self._model

    @property
    def generator(self):
        """Lazy load constrained generator."""
        if self._generator is None:
            self._generator = outlines.generate.json(self.model, EXTRACTION_JSON_SCHEMA)
        return self._generator

    def extract(self, trial_text: str, max_tokens: int = 512) -> dict[str, Any]:
        """Extract structured fields with constrained decoding."""
        prompt = create_extraction_prompt(trial_text)

        try:
            result = self.generator(prompt, max_tokens=max_tokens, sampler=greedy())
        except Exception:
            logger.exception("constrained_extraction_failed")
            return self._fallback_extract(trial_text)
        else:
            return result

    def _fallback_extract(self, trial_text: str) -> dict[str, Any]:
        """Fallback extraction without constraints."""
        create_extraction_prompt(trial_text)
        return {
            "primary_endpoint": "",
            "endpoint_category": "other",
            "intervention_names": [],
            "intervention_class": "other",
            "randomized": False,
            "blinding": "not_applicable",
            "population_age_group": "adult",
            "key_inclusion": [],
            "key_exclusion": [],
        }


class LoRAExtractor:
    """LoRA fine-tuned extractor for trial data."""

    def __init__(self, model_path: str = "models/qwen2.5-3b-lora-extractor", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load fine-tuned LoRA model."""
        if self._model is None:
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer

                base_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-3B-Instruct",
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True,
                )

                self._model = PeftModel.from_pretrained(base_model, self.model_path)
                self._model = self._model.merge_and_unload()
                self._model.eval()

                self._tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-3B-Instruct",
                    trust_remote_code=True,
                    padding_side="left",
                )
                self._tokenizer.pad_token = self._tokenizer.eos_token

            except Exception:
                logger.exception("lora_model_load_failed")
                raise

    def extract(
        self, trial_text: str, max_new_tokens: int = 512, temperature: float = 0.0
    ) -> dict[str, Any]:
        """Extract using fine-tuned LoRA model."""
        self._load_model()

        prompt = create_extraction_prompt(trial_text)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=800,
        ).to(self._model.device)

        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1] :]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("lora_json_parse_failed", response=response[:200])
            return self._extract_json_from_text(response)

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """Try to extract valid JSON from text."""
        import re

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}


def extract_trial_fields(
    trial_text: str,
    method: str = "lora",
    model_path: str | None = None,
) -> dict[str, Any]:
    """Main entry point for trial field extraction."""
    if method == "constrained":
        if not OUTLINES_AVAILABLE:
            raise RuntimeError("Outlines not installed. Run: uv add outlines")
        extractor = ConstrainedExtractor()
        return extractor.extract(trial_text)
    elif method == "lora":
        extractor = LoRAExtractor(model_path or "models/qwen2.5-3b-lora-extractor")
        return extractor.extract(trial_text)
    else:
        raise ValueError(f"Unknown extraction method: {method}")
