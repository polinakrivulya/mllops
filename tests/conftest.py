import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from src.utils.config import load_config
from src.prepare import main as prepare_main


def _run_main(main_func, argv):
    old_argv = sys.argv
    try:
        sys.argv = argv
        main_func()
    finally:
        sys.argv = old_argv


@pytest.fixture(scope="session")
def prepared_cfg_path(tmp_path_factory) -> str:
    """
    Готовит processed dataset в tmp директории один раз на всю сессию тестов.
    Возвращает путь до временного yaml-конфига.
    """
    base_cfg = load_config("configs/tiny-fast.yaml")

    workdir = tmp_path_factory.mktemp("prepared_data")
    raw_dir = workdir / "data" / "raw" / "emotion"
    processed_dir = workdir / "data" / "processed" / "emotion"

    base_cfg["data"]["raw_dir"] = str(raw_dir)
    base_cfg["data"]["processed_dir"] = str(processed_dir)

    # Чуть ускорим prepare (если вы добавили max_* в prepare — отлично; если нет, можно не трогать)
    base_cfg["data"]["max_train_samples"] = base_cfg["data"].get("max_train_samples", 64)
    base_cfg["data"]["max_eval_samples"] = base_cfg["data"].get("max_eval_samples", 64)

    cfg_path = workdir / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, allow_unicode=True)

    # Запускаем prepare (скачает датасет и сохранит processed_dir)
    _run_main(prepare_main, ["prepare", "--config", str(cfg_path)])

    assert Path(base_cfg["data"]["processed_dir"]).exists()

    return str(cfg_path)
