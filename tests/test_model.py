import torch

from src.utils.config import load_config
from src.data import load_data_and_tokenizer
from src.model import build_model

def test_forward_shapes_with_collator():
    cfg = load_config("configs/tiny-fast.yaml")
    ds, tokenizer, id2label, label2id, num_labels = load_data_and_tokenizer(cfg)
    model = build_model(cfg, num_labels, id2label, label2id)

    # Берём несколько примеров и паддим через collator (иначе длины разные)
    examples = [ds["train"][i] for i in range(4)]
    batch = ds["collator"](examples)

    input_ids = batch["input_ids"]  # уже torch.Tensor
    attention_mask = batch["attention_mask"]

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (input_ids.shape[0], num_labels)
