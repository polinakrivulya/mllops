import pytest

from src.utils.config import load_config
from src.data import load_data_and_tokenizer


def test_dataset_schema_and_ranges():
    cfg = load_config("configs/tiny-fast.yaml")
    ds, tokenizer, id2label, label2id, num_labels = load_data_and_tokenizer(cfg)

    # Базовые ключи и размеры
    assert isinstance(ds, dict)
    assert "train" in ds and "eval" in ds and "collator" in ds
    assert len(ds["train"]) > 0
    assert len(ds["eval"]) > 0
    assert num_labels == len(id2label) == len(label2id)

    # структура и допустимые диапазоны
    sample = ds["train"][0]
    for k in ("input_ids", "attention_mask", "labels"):
        assert k in sample

    assert isinstance(sample["labels"], int)
    assert 0 <= sample["labels"] < num_labels

    # Проверка длины токенов
    max_len = int(cfg["data"]["max_seq_length"])
    assert isinstance(sample["input_ids"], list)
    assert len(sample["input_ids"]) <= max_len

    for i in range(num_labels):
        assert i in id2label
        assert id2label[i] in label2id
        assert label2id[id2label[i]] == i
