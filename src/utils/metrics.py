from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import evaluate 
from transformers.trainer_utils import EvalPrediction

def compute_metrics_builder(id2label: Optional[Dict[int, str]] = None) -> Callable[[Union[EvalPrediction, Tuple[np.ndarray, np.ndarray]]], Dict[str, float]]:
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    def compute(eval_pred: Union[EvalPrediction, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        if isinstance(eval_pred, tuple):
            logits, labels = eval_pred
        else:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        metrics = {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

        # id2label оставлен для будущего: можно добавить метрики по классам с красивыми именами
        # if id2label:
        #     from sklearn.metrics import f1_score
        #     labels_order = sorted(id2label.keys())
        #     per_class = f1_score(labels, preds, average=None, labels=labels_order)
        #     for i, score in zip(labels_order, per_class):
        #         metrics[f"f1_{id2label[i]}"] = float(score)
        return metrics
    return compute
