from __future__ import annotations
from typing import Any, Dict, List, Optional


def build_preprocess_config(
    tokenizer_name: str,
    text_field: str = "text",
    max_length: int = 128,
    lower_case: bool = False,
) -> Dict[str, Any]:
    return {
        "tokenizer_name": tokenizer_name,
        "text_field": text_field,
        "max_length": int(max_length),
        "lower_case": bool(lower_case),
    }


def load_and_preprocess(dataset_name: str, split: str, config: Dict[str, Any], streaming: bool = False, **kwargs):
    from .hf import load_dataset, preprocess_examples
    ds = load_dataset(dataset_name, split=split, streaming=streaming, **kwargs)
    return preprocess_examples(ds, config["tokenizer_name"], text_field=config.get("text_field", "text"), max_length=config.get("max_length", 128), lower_case=config.get("lower_case", False))


