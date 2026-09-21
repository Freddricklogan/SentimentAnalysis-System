# API container: docker build -t sentiment-system . && docker run --rm -p 8000:8000 sentiment-system
# The model (268 MB) is downloaded at build time so the container starts offline.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf-cache
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu . \
 && python -c "from sentiment_system import model; model.classify(['warm up'])" \
 && useradd -m app && chown -R app /app
USER app
EXPOSE 8000
CMD ["sentiment-system", "serve", "--host", "0.0.0.0", "--port", "8000"]
