FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY README.md .

RUN pip install --no-cache-dir -e .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/viz/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
