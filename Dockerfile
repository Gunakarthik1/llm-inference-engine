FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY engine/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY engine/ /app/engine/
COPY tests/ /app/tests/

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "engine.main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
