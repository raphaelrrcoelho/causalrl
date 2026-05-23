# Multi-stage build using the official uv image.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base
WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
RUN uv sync --extra dev --no-install-project
RUN uv sync --extra dev

# Default: run the test suite
CMD ["uv", "run", "pytest", "-v"]
