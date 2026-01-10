FROM python:3.11-slim

WORKDIR /app

# системные зависимости (часто нужны pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# исходный код
COPY src /app/src
COPY configs /app/configs

# сохранённая модель (должна существовать локально перед docker build)
COPY models/bert-tiny /app/models/bert-tiny

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.predict"]
