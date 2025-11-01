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

**Данные:**
- Hugging Face Datasets: dair-ai/emotion
- Преимущества: легальная доступность (не NDA), небольшой размер, готовая разметка, быстрые загрузка/предобработка

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
