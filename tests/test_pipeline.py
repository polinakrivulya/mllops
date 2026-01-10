import sys
from pathlib import Path

import yaml

from src.utils.config import load_config
from src.train import main as train_main


def _run_main(main_func, argv):
    old_argv = sys.argv
    try:
        sys.argv = argv
        main_func()
    finally:
        sys.argv = old_argv


def test_end_to_end(tmp_path, prepared_cfg_path):
    cfg = load_config(prepared_cfg_path)

    out_dir = tmp_path / "model"
    cfg["training"]["output_dir"] = str(out_dir)

    # Чтобы тесты не падали на сохранении/чекпоинтах
    cfg["training"]["save_strategy"] = "no"
    cfg["training"]["load_best_model_at_end"] = False
    cfg["training"]["save_safetensors"] = False

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    _run_main(train_main, ["train", "--config", str(cfg_path)])

    assert (out_dir / "config.json").exists()
    assert (out_dir / "pytorch_model.bin").exists() or (out_dir / "model.safetensors").exists()
    assert (out_dir / "tokenizer_config.json").exists()
