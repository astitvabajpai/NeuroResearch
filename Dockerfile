FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data directory for SQLite
RUN mkdir -p /data && chmod 777 /data

# HF Spaces runs on port 7860
EXPOSE 7860

# Start server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
