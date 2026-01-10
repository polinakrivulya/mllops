from typing import Any, Dict, Tuple

from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding


def load_data_and_tokenizer(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Any, Dict[int, str], Dict[str, int], int]:
    processed_dir = cfg["data"]["processed_dir"]
    ds = load_from_disk(processed_dir)

    label_col = cfg["data"]["label_column"]  # обычно "label"

    labels = ds["train"].features[label_col].names
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name_or_path"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = ds["train"]
    eval_split = cfg["runtime"].get("eval_on", "validation")
    eval_ds = ds[eval_split]

    num_labels = len(labels)

    return (
        {"train": train_ds, "eval": eval_ds, "collator": collator},
        tokenizer,
        id2label,
        label2id,
        num_labels,
    )
