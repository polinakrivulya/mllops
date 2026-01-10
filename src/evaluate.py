import argparse
import json
import logging
import os
from typing import Any, Dict

import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from src.utils.config import load_config
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    logger = get_logger("evaluate", logging.DEBUG if args.verbose else logging.INFO)

    processed_dir = cfg["data"]["processed_dir"]
    model_dir = cfg["training"]["output_dir"]
    metrics_path = cfg["evaluate"]["metrics_path"]

    ds = load_from_disk(processed_dir)
    eval_split = cfg["runtime"].get("eval_on", "validation")
    eval_ds = ds[eval_split]

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Running evaluation")
    metrics = trainer.evaluate(eval_dataset=eval_ds)

    out = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    # ВАЖНО: создать директорию под метрики
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    logger.info(f"Saving metrics to reports/metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(out)


if __name__ == "__main__":
    main()
