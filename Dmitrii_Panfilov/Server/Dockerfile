# Dockerfile for chatGPT and/or other LLM
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Установка необходимых пакетов и обновление системы
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    pip install --no-cache-dir -U pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip with no cache
# RUN pip install --no-cache-dir -U pip

# рабочая директория в контейнере
WORKDIR /app

# Копирование файлов в рабочую директорию
COPY . /app/

# Установка дополнительных зависимостей, указанных в requirements.txt (если они есть)
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Запуск FastAPI на порту 8000 на 0.0.0.0 (по умолчанию) и слушать на этом порту
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
