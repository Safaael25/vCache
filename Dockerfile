FROM python:3.11-slim

ENV POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="/opt/poetry/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copy dependency files first so the dependency layer is cached across code changes.
COPY pyproject.toml poetry.lock README.md ./

# Same groups as .github/workflows/ci.yml, plus "benchmarks" for running benchmarks/benchmark.py.
RUN poetry install --with dev,benchmarks --no-root

COPY . .

RUN poetry install --with dev,benchmarks

ENTRYPOINT ["poetry", "run"]
CMD ["pytest", "tests/unit", "--import-mode=importlib"]
