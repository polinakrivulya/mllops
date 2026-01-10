## Классификация эмоций в коротких текстах

**Бизнес-цель**: Автоматическая маршрутизация обращений пользователей в поддержку: приоритетная обработка негативных эмоций (anger, sadness, fear) для их приоритезации и снижения среднего времени решения.

**Продакшн-метрики**:
- Качество:
    - Macro F1 на валидации ≥ 0.85 (целевой минимум для проекта)
    - Accuracy ≥ 0.85 (вторичная)
- Технические:
    - P95 latency ≤ 150 мс на CPU (при batch_size=1 и последовательной обработке)
    - Доля ошибок инференса ≤ 1%
    - RAM ≤ 250 МБ
- Надежность обучения:
    - Фиксированный random_seed
    - Воспроизводимость результатов ±0.5% F1 (при одном и том же seed)

## Датасет
Источник: Hugging Face Datasets — `dair-ai/emotion`  
https://huggingface.co/datasets/dair-ai/emotion

Датасет автоматически скачивается на стадии `prepare` и сохраняется локально (см. ниже).

## Где физически лежат данные/модели (DVC-артефакты)
Большие файлы **не хранятся в Git**, но версионируются через **DVC**:

- **Raw dataset**: `data/raw/emotion/`
- **Processed dataset**: `data/processed/emotion/`
- **Trained model (HF format)**: `models/bert-tiny/`
- **Metrics**: `reports/metrics.json`

Все эти артефакты можно восстановить командой `dvc pull` для любой версии репозитория (после `git checkout <commit>`).

**План экспериментов:**
- Бейзлайн: majority class
- Простой классический бейзлайн: TF-IDF + LogisticRegression
- Нейросетевые:
    - «prajjwal1/bert-tiny»
    - «distilbert-base-uncased» (как улучшение качества — возможно)
- Варианты:
    - Полный fine-tuning vs частичное замораживание слоев
    - Подбор lr, max_seq_len
- Выбор лучшей модели по F1-macro на валидации

**Установка и запуск**
- Установка:
  
((Нужен python 3.11))

brew install python@3.11

/opt/homebrew/bin/python3.11 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt

- Обучение:
python -m src.train --config configs/default.yaml --verbose
- Быстрая проверка (маленький прогон):
python -m src.train --config configs/tiny-fast.yaml --verbose
- Оценка сохраненной модели:
python -m src.eval --model_dir outputs/bert-tiny --split validation

## MLflow tracking

Каждый запуск `python -m src.train --config ...` создаёт отдельный MLflow run с:
- параметрами (весь YAML конфиг),
- метриками (eval_f1, eval_accuracy, eval_loss),
- артефактами (HF-модель, конфиг запуска, dvc.lock).

## Docker

### Что делает контейнер
Контейнер запускает `python -m src.predict` и выполняет batch-inference:
- читает CSV из `--input_path` (ожидается колонка `text`)
- загружает модель из `--model_dir` (по умолчанию `models/bert-tiny`)
- записывает CSV в `--output_path` с колонками:
  - `pred_id` (int)
  - `pred_label` (str)
  - `pred_score` (float)

### Build
Перед сборкой убедитесь, что модель существует локально: `models/bert-tiny/`.

```bash
docker build -t ml-app:v1 .