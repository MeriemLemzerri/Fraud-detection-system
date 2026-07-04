# Build context is the repository root (see docker-compose.yml) so this
# image can also bundle assets/ and sample_test.csv produced by
# train_pipeline.py, which the "Model Performance" and "Single Transaction
# Simulator" tabs read.
FROM python:3.10-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./
COPY assets/ ./assets/
COPY sample_test.csv ./sample_test.csv

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
