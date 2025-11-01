import argparse
import os
import logging
import numpy as np
import random
import torch
import evaluate
from transformers import TrainingArguments, Trainer
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data import load_data_and_tokenizer
from src.model import build_model
from src.utils.metrics import compute_metrics_builder

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

    cfg = load_config(args.config)
    logger = get_logger(
        "train",
        logging.DEBUG if args.verbose else logging.INFO,
    )

    set_seed(int(cfg["seed"]))

    ds, tokenizer, id2label, label2id, num_labels = load_data_and_tokenizer(cfg)
    model = build_model(cfg, num_labels, id2label, label2id)

    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        num_train_epochs=float(cfg["training"]["num_train_epochs"]),
        per_device_train_batch_size=int(
            cfg["training"]["per_device_train_batch_size"]
        ),
        per_device_eval_batch_size=int(
            cfg["training"]["per_device_eval_batch_size"]
        ),
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        logging_steps=int(cfg["training"]["logging_steps"]),
        evaluation_strategy=cfg["training"].get("evaluation_strategy", "epoch"),
        save_strategy=cfg["training"].get("save_strategy", "epoch"),
        load_best_model_at_end=cfg["training"].get("load_best_model_at_end", True),
        metric_for_best_model=cfg["training"].get("metric_for_best_model", "f1"),
        greater_is_better=cfg["training"].get("greater_is_better", True),
        report_to=cfg["runtime"].get("report_to", "none"),
        fp16=bool(cfg["training"].get("fp16", False)),
        save_safetensors=bool(cfg["training"].get("save_safetensors", False))
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        tokenizer=tokenizer,
        data_collator=ds["collator"],
        compute_metrics=compute_metrics_builder(id2label),
    )

    logger.info("Start training")
    trainer.train()
    logger.info("Training finished")

    eval_metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_metrics}")

    logger.info("Saving model and tokenizer (save_pretrained)")
    model.save_pretrained(cfg["training"]["output_dir"], safe_serialization=bool(cfg["training"].get("save_safetensors", False)))
    tokenizer.save_pretrained(cfg["training"]["output_dir"])

if __name__ == "__main__":
    main()
