FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

# Install deps as a cacheable layer before copying source.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

ENV VININATOR_DATA_DIR=/data

CMD ["uv", "run", "vininator", "features", "climate"]
