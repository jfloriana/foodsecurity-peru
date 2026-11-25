# FoodSecurity AI V2 - imagen de la aplicación Streamlit y del pipeline de entrenamiento
# El contexto de compilación es la RAÍZ del proyecto (docker-compose.yml: context "..", dockerfile "app_streamlit/Dockerfile"):
#   docker build -f app_streamlit/Dockerfile .
# Disposición en la imagen: /app/fsai (biblioteca), /app/app (Streamlit), /app/train_pipeline.py, /app/resultados_agregados.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    FSAI_RAIZ_PROYECTO=/app     FSAI_RESULTADOS_AGREGADOS=/app/resultados_agregados \
    FSAI_ARTIFACTS_DIR=/app/artifacts \
    HOME=/home/fsai

# libgomp1: runtime OpenMP que necesitan xgboost/catboost/scikit-learn en la imagen slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1     && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios (UID fijo para permisos de volúmenes)
RUN groupadd --gid 10001 fsai \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin fsai

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas. El lock fija versiones exactas y trae el índice CPU de PyTorch
# (--extra-index-url), de modo que torch se instala sin CUDA.
COPY app_streamlit/requirements-lock.txt ./requirements-lock.txt
RUN pip install -r requirements-lock.txt

COPY --chown=fsai:fsai foodsecurity_ai/fsai/ ./fsai/
COPY --chown=fsai:fsai foodsecurity_ai/train_pipeline.py ./train_pipeline.py
COPY --chown=fsai:fsai app_streamlit/app/ ./app/
COPY --chown=fsai:fsai resultados_agregados/ ./resultados_agregados/
RUN mkdir -p /app/artifacts /app/data && chown fsai:fsai /app/artifacts /app/data

USER fsai
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/app.py"]
