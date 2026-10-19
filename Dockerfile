FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TEMP_DIR=/app/temp

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 ocrbot
WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY app ./app
COPY tests ./tests
COPY pytest.ini .env.example ./
RUN mkdir -p /app/temp /app/data && chown -R ocrbot:ocrbot /app

USER ocrbot
# Модели PaddleOCR скачиваются при первом запуске в ~/.paddlex — вынесите в volume.
VOLUME ["/home/ocrbot/.paddlex"]

CMD ["python", "-m", "app.bot"]
