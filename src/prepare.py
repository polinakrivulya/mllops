import argparse
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from src.utils.config import load_config
from src.utils.logger import get_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    logger = get_logger("prepare", logging.DEBUG if args.verbose else logging.INFO)

    set_seed(int(cfg["seed"]))

    dataset_name = cfg["data"]["dataset_name"]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    max_len = int(cfg["data"]["max_seq_length"])

    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    logger.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    # Если в датасете нет validation — создаём
    if "validation" not in ds:
        tmp = ds["train"].train_test_split(test_size=0.1, seed=int(cfg["seed"]))
        ds = DatasetDict({"train": tmp["train"], "validation": tmp["test"]})

    logger.info(f"Saving RAW dataset to: {raw_dir}")
    ds.save_to_disk(raw_dir)

    logger.info("Tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name_or_path"])

    def tokenize(batch):
        tok = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_len,
        )
        # Trainer ожидает "labels"
        tok["labels"] = batch[label_col]
        return tok

    tokenized = ds.map(tokenize, batched=True, remove_columns=[text_col])

    logger.info(f"Saving PROCESSED dataset to: {processed_dir}")
    tokenized.save_to_disk(processed_dir)

    logger.info("Done")


if __name__ == "__main__":
    main()
