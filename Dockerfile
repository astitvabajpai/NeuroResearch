FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data directory for SQLite persistence
RUN mkdir -p /data && chmod 777 /data

# HF Spaces mounts persistent storage at /data automatically
# The DB_PATH in database.py detects /data and uses it when writable

# HF Spaces runs on port 7860
EXPOSE 7860

# Use app.py entrypoint (standard for HF Spaces)
CMD ["python", "app.py"]
