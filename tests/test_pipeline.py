import os
import sys

import yaml

from src.utils.config import load_config
from src.train import main as train_main


def test_end_to_end(tmp_path, monkeypatch):
    # Загружаем базовый конфиг и перенаправляем вывод в tmp
    cfg = load_config("configs/tiny-fast.yaml")
    out_dir = tmp_path / "model"

    # Делает прогон максимально быстрым и безопасным для CI
    cfg["training"]["output_dir"] = str(out_dir)
    cfg["training"]["save_strategy"] = "no"            # без чекпоинтов
    cfg["training"]["load_best_model_at_end"] = False  # чтобы не искать лучший чекпоинт
    cfg["training"]["save_safetensors"] = False        # сохранять в .bin, чтобы избежать non-contiguous ошибок

    # Временный конфиг
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    # Подавляем ворнинги в тестах
    monkeypatch.setenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")

    # Эмулируем запуск через CLI
    sys.argv = ["train", "--config", str(cfg_path)]
    train_main()

    # Проверяем, что модель и токенизатор сохранены в формате Hugging Face
    assert (out_dir / "config.json").exists()
    # Для PyTorch-сейва
    assert (out_dir / "pytorch_model.bin").exists()
    # Токенайзер
    assert (out_dir / "tokenizer_config.json").exists()
