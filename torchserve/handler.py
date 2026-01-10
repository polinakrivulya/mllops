import json
from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class EmotionHandler(BaseHandler):
    """
    Вход:
      - JSON: {"text": "I am happy"} или {"texts": ["...", "..."]}
    Выход:
      - список объектов: [{"label": "...", "score": 0.93}, ...]
    """

    def __init__(self):
        super().__init__()
        self.device = None
        self.model = None
        self.tokenizer = None
        self.id2label: Dict[int, str] = {}
        self.initialized = False

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = properties.get("model_dir")
        # serializedFile из MAR будет доступен как model.pt в model_dir
        state_path = f"{model_dir}/model.pt"

        # В extra-files мы положим config/tokenizer файлы, поэтому грузим "from_pretrained(model_dir)"
        config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = AutoModelForSequenceClassification.from_config(config)

        state_dict = torch.load(state_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # id2label может иметь строковые ключи
        cfg_id2label = getattr(self.model.config, "id2label", None)
        if isinstance(cfg_id2label, dict):
            self.id2label = {int(k): v for k, v in cfg_id2label.items()}
        else:
            # fallback
            self.id2label = {i: str(i) for i in range(self.model.config.num_labels)}

        self.initialized = True

    def preprocess(self, data):
        # TorchServe передает список запросов; каждый элемент содержит "body" или "data"
        texts: List[str] = []

        for row in data:
            raw = row.get("body") if isinstance(row, dict) else row
            if raw is None:
                raw = row.get("data") if isinstance(row, dict) else raw

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")

            if isinstance(raw, str):
                payload = json.loads(raw)
            elif isinstance(raw, dict):
                payload = raw
            else:
                raise ValueError("Unsupported input format")

            if "text" in payload:
                texts.append(str(payload["text"]))
            elif "texts" in payload:
                texts.extend([str(t) for t in payload["texts"]])
            else:
                raise ValueError('JSON must contain key "text" or "texts"')

        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc, texts

    def inference(self, data, *args, **kwargs):
        enc, texts = data
        with torch.no_grad():
            out = self.model(**enc)
        logits = out.logits.detach().cpu().numpy()
        probs = _softmax(logits, axis=-1)
        pred_ids = probs.argmax(axis=-1)
        pred_scores = probs.max(axis=-1)
        return pred_ids, pred_scores

    def postprocess(self, inference_output):
        pred_ids, pred_scores = inference_output
        result = []
        for i, s in zip(pred_ids.tolist(), pred_scores.tolist()):
            result.append({"label": self.id2label[int(i)], "score": float(s)})
        return result
