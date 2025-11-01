from typing import Dict
from transformers import AutoConfig, AutoModelForSequenceClassification, BertConfig, PreTrainedModel

def build_model(cfg: Dict, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]) -> PreTrainedModel:
    init_from_scratch = bool(cfg.get("model", {}).get("init_from_scratch", False))
    if init_from_scratch:
        base = "bert-base-uncased"
        config = BertConfig.from_pretrained(
            base,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
        )
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model_name = cfg["model"]["model_name_or_path"]
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

    return model
