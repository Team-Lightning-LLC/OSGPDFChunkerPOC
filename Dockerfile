FROM python:3.11-slim

# Install Tesseract OCR (for fallback on PDFs without text layers)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

# Single worker (in-memory job store), long timeout for big PDFs
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "--workers", "1", "app:app"]
