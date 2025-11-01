import argparse
import logging
from typing import Callable, Dict, Tuple, Union
import numpy as np
import evaluate 
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer
from transformers.trainer_utils import EvalPrediction
from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics_builder

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = get_logger(
        "eval",
        logging.DEBUG if args.verbose else logging.INFO,
    )
    # Датасет
    ds = load_dataset("dair-ai/emotion")
    if args.split in ds:
        data = ds[args.split]
    else:
        tmp = ds["train"].train_test_split(test_size=0.1, seed=42)
        data = tmp["test"] if args.split != "train" else tmp["train"]

    if args.max_samples:
        n = min(len(data), int(args.max_samples))
        data = data.select(range(n))

    # Модель и токенизатор
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    def tokenize(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
        )
        tok["labels"] = batch["label"]
        return tok

    data = data.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in data.column_names if c in ("text", "label")],
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    cfg_id2label = getattr(model.config, "id2label", None)
    if isinstance(cfg_id2label, dict):
        id2label = {int(k): v for k, v in cfg_id2label.items()}
    else:
        id2label = None
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    logger.info("Start evaluation")
    metrics = trainer.evaluate(eval_dataset=data)
    logger.info(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()
