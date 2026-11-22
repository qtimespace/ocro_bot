FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TEMP_DIR=/app/temp

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 fonts-dejavu-core \
       libenchant-2-2 hunspell-ru hunspell-en-us \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 ocrbot
WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY app ./app
# Модели OCR (ONNX, ~20 МБ) скачиваются при сборке: бот стартует без сети к хостингам моделей.
RUN python -c "from app.services.ocr_engine import RapidOcrEngine; RapidOcrEngine(['ru', 'en']).warmup()"
COPY tests ./tests
COPY pytest.ini .env.example ./
# Каталоги заранее с владельцем ocrbot: Docker копирует права в новые тома при первом подключении.
RUN mkdir -p /app/temp /app/data /home/ocrbot/.paddlex \
    && chown -R ocrbot:ocrbot /app /home/ocrbot/.paddlex

USER ocrbot
# Кэш моделей PaddleOCR — нужен только при OCR_BACKEND=paddle.
VOLUME ["/home/ocrbot/.paddlex"]

CMD ["python", "-m", "app.bot"]
