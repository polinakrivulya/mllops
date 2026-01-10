import argparse
import logging
import os
import random
from typing import Any, Dict

import evaluate
import mlflow
import mlflow.transformers
import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from src.data import load_data_and_tokenizer
from src.model import build_model
from src.utils.config import load_config
from src.utils.logger import get_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def compute_metrics_builder():
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    return compute


def try_log_artifact(path: str) -> None:
    if os.path.exists(path):
        mlflow.log_artifact(path)


def try_log_artifacts_dir(path: str, artifact_path: str) -> None:
    if os.path.isdir(path):
        mlflow.log_artifacts(path, artifact_path=artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    logger = get_logger("train", logging.DEBUG if args.verbose else logging.INFO)

    set_seed(int(cfg["seed"]))

    # --- MLflow setup ---
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name", "default")
    run_name = mlflow_cfg.get("run_name")
    autolog = bool(mlflow_cfg.get("autolog", True))

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    if autolog:
        # Для transformers умеет логировать параметры/метрики/модель (в зависимости от версии)
        mlflow.transformers.autolog(log_models=False)

    # -----------------------------------------
    with mlflow.start_run(run_name=run_name):
        # логируем параметры (конфиг целиком, в плоском виде)
        flat_cfg = flatten_dict(cfg)
        mlflow.log_params({k: str(v) for k, v in flat_cfg.items()})

        # полезные теги (опционально)
        mlflow.set_tag("config_path", args.config)

        ds, tokenizer, id2label, label2id, num_labels = load_data_and_tokenizer(cfg)
        model = build_model(cfg, num_labels, id2label, label2id)

        output_dir = cfg["training"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            num_train_epochs=float(cfg["training"]["num_train_epochs"]),
            per_device_train_batch_size=int(cfg["training"]["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(cfg["training"]["per_device_eval_batch_size"]),
            warmup_ratio=float(cfg["training"]["warmup_ratio"]),
            logging_steps=int(cfg["training"]["logging_steps"]),
            evaluation_strategy=cfg["training"].get("evaluation_strategy", "epoch"),
            save_strategy=cfg["training"].get("save_strategy", "epoch"),
            load_best_model_at_end=cfg["training"].get("load_best_model_at_end", True),
            metric_for_best_model=cfg["training"].get("metric_for_best_model", "f1"),
            greater_is_better=cfg["training"].get("greater_is_better", True),
            report_to=cfg["runtime"].get("report_to", "none"),
            fp16=bool(cfg["training"].get("fp16", False)),
            save_safetensors=bool(cfg["training"].get("save_safetensors", False)),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["eval"],
            tokenizer=tokenizer,
            data_collator=ds["collator"],
            compute_metrics=compute_metrics_builder(),
        )

        logger.info("Start training")
        trainer.train()
        logger.info("Training finished")

        eval_metrics = trainer.evaluate()
        logger.info(f"Validation metrics: {eval_metrics}")

        # логируем метрики вручную (чтобы точно появились в MLflow)
        mlflow.log_metrics(
            {
                "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
                "eval_accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
                "eval_f1": float(eval_metrics.get("eval_f1", 0.0)),
            }
        )

        # сохраняем модель и токенизатор
        logger.info("Saving model and tokenizer (save_pretrained)")
        model.save_pretrained(
            output_dir,
            safe_serialization=bool(cfg["training"].get("save_safetensors", False)),
        )
        tokenizer.save_pretrained(output_dir)

        # --- Артефакты в MLflow ---
        # 1) модель как директория артефактов (HF формат)
        try_log_artifacts_dir(output_dir, artifact_path="model")

        # 2) конфиг запуска
        try_log_artifact(args.config)

        # 3) dvc.lock (опционально, если есть)
        try_log_artifact("dvc.lock")


if __name__ == "__main__":
    main()
