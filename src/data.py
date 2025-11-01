from typing import Any, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def load_data_and_tokenizer(cfg) -> Tuple[Dict, Any, Dict[int, str], Dict[str, int], int]:
    ds = load_dataset(cfg["data"]["dataset_name"])
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    labels = ds["train"].features[label_col].names
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_name_or_path"]
    )
    max_len = int(cfg["data"]["max_seq_length"])

    def tokenize(batch):
        tok = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_len,
        )
        tok["labels"] = batch[label_col]
        return tok

    ds = ds.map(tokenize, batched=True, remove_columns=[text_col])


    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    num_labels = len(labels)

    return (
        {"train": train_ds, "eval": eval_ds, "collator": collator},
        tokenizer,
        id2label,
        label2id,
        num_labels,
    )
