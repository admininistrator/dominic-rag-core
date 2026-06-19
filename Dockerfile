FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY src ./src
COPY startup.sh ./startup.sh
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        'fastapi>=0.109,<0.111' \
        'uvicorn>=0.27,<0.30' \
        'gunicorn>=22,<23' \
        'httpx>=0.27,<0.28' \
        'qdrant-client>=1.13,<1.15' \
        'sqlalchemy>=2.0,<3.0' \
        'alembic>=1.13,<2.0' \
        'psycopg[binary]>=3.1,<4.0' \
        'boto3>=1.34,<2.0' \
        'python-multipart>=0.0.9,<1.0' \
        'celery[redis]>=5.3,<6.0' \
        'redis>=5.0,<6.0' \
    && pip install --no-deps . \
    && chmod +x ./startup.sh

EXPOSE 8010

CMD ["./startup.sh"]
