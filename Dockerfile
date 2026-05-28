FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY src ./src
COPY startup.sh ./startup.sh

RUN pip install --upgrade pip \
    && pip install . \
    && chmod +x ./startup.sh

EXPOSE 8010

CMD ["./startup.sh"]
