from src.utils.config import load_config
from src.data import load_data_and_tokenizer
from src.model import build_model


def test_forward_shapes_with_collator(prepared_cfg_path):
    cfg = load_config(prepared_cfg_path)
    ds, tokenizer, id2label, label2id, num_labels = load_data_and_tokenizer(cfg)
    model = build_model(cfg, num_labels, id2label, label2id)

    examples = [ds["train"][i] for i in range(4)]
    batch = ds["collator"](examples)

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    assert out.logits.shape[0] == 4
    assert out.logits.shape[1] == num_labels
