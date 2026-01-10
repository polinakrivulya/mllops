from src.utils.config import load_config
from src.data import load_data_and_tokenizer


def test_dataset_schema_and_ranges(prepared_cfg_path):
    cfg = load_config(prepared_cfg_path)
    ds, tokenizer, id2label, label2id, num_labels = load_data_and_tokenizer(cfg)

    assert "train" in ds and "eval" in ds and "collator" in ds
    assert len(ds["train"]) > 0
    assert len(ds["eval"]) > 0
    assert num_labels == len(id2label) == len(label2id)

    sample = ds["train"][0]
    for k in ("input_ids", "attention_mask", "labels"):
        assert k in sample

    assert isinstance(sample["labels"], int)
    assert 0 <= sample["labels"] < num_labels

    max_len = int(cfg["data"]["max_seq_length"])
    assert isinstance(sample["input_ids"], list)
    assert len(sample["input_ids"]) <= max_len
