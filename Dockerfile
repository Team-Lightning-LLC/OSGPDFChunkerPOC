FROM python:3.11-slim

# Install Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8080

# Run with gunicorn (production server)
# Single worker required for in-memory job storage
# Long timeout for big PDFs
# Internal parallelism handled by ThreadPoolExecutor
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "--workers", "1", "app:app"]
