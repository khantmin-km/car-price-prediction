# 3.11 rather than a newer interpreter: the pinned scikit-learn publishes cp311
# wheels as its newest, so 3.12+ would compile it from source on every build.
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_CACHE=1

WORKDIR /app

# Dependencies resolve in their own layer so they are only reinstalled when the
# lock file changes, not on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Installing the project itself puts `src` on the import path regardless of
# working directory.
COPY src/ ./src/
RUN uv sync --frozen --no-dev


FROM python:3.11-slim-bookworm

# The environment is copied rather than rebuilt, leaving uv and its downloads
# out of the runtime image.
WORKDIR /app
COPY --from=builder /app/.venv ./.venv

COPY src/ ./src/
COPY app/code/ ./code/

# PYTHONPATH covers the app modules, which import each other by bare name.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/code \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8050

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8050') + '/')"

CMD ["python", "code/main.py"]
