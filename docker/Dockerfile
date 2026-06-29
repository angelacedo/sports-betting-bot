FROM python:3.11-slim-bookworm

LABEL maintainer="sports-betting-bot"
LABEL description="Sports Betting Bot - Production"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appgroup \
    && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sql/ ./sql/

RUN mkdir -p /app/logs /app/data \
    && chown -R appuser:appgroup /app

USER appuser

CMD ["python", "-m", "src.scheduler"]
