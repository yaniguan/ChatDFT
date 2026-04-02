FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev openssh-client rsync && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create non-root user
RUN useradd -m chatdft && chown -R chatdft:chatdft /app
USER chatdft

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); assert r.status_code==200"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
