FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/tableagent

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        fonts-dejavu-core \
        fontconfig \
        libreoffice-calc \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml README.md ./
COPY TableAgent ./TableAgent
COPY service ./service
COPY config.example.yaml ./config.example.yaml

RUN python -m pip install --no-cache-dir . \
    && mkdir -p /tmp/tableagent cache outputs

EXPOSE 3636

CMD ["table-agent-api", "--config", "/app/config.example.yaml", "--host", "0.0.0.0", "--port", "3636"]
