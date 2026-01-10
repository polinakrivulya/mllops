import argparse
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logger import get_logger


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def batch_iter(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/bert-tiny")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = get_logger("predict", logging.DEBUG if args.verbose else logging.INFO)

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    logger.info(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    df = pd.read_csv(args.input_path)
    if args.text_column not in df.columns:
        raise ValueError(
            f"Input CSV must contain column '{args.text_column}'. Got columns: {list(df.columns)}"
        )

    texts = df[args.text_column].astype(str).tolist()
    all_pred_ids = []
    all_pred_scores = []

    device = torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        for batch_texts in batch_iter(texts, args.batch_size):
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()
            probs = softmax(logits, axis=-1)
            pred_ids = probs.argmax(axis=-1)
            pred_scores = probs.max(axis=-1)

            all_pred_ids.extend(pred_ids.tolist())
            all_pred_scores.extend(pred_scores.tolist())

    id2label = model.config.id2label
    # иногда ключи в config могут быть строками
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}
    pred_labels = [id2label[int(i)] for i in all_pred_ids]

    out_df = df.copy()
    out_df["pred_id"] = all_pred_ids
    out_df["pred_label"] = pred_labels
    out_df["pred_score"] = all_pred_scores

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_df.to_csv(args.output_path, index=False)

    logger.info(f"Saved predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
