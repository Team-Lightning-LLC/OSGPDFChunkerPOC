# PDF Chunker — OCR Edition

Detects customer boundaries in multi-customer After Sample PDFs using OCR. Splits into batch files for parallel processing.

## Endpoints

### `GET /` 
Web UI for testing — upload a PDF, see detected customers.

### `POST /process`
Analyze PDF and return manifest (no splitting).
- **Body**: `multipart/form-data` with `file` (PDF) and optional `batch_size` (default 12)
- **Returns**: JSON with manifest, customers, page texts

### `POST /split`
Analyze PDF, split into batch files, return as zip.
- **Body**: `multipart/form-data` with `file` (PDF) and optional `batch_size` (default 12)
- **Returns**: ZIP file containing `manifest.json` + `batch_001.pdf`, `batch_002.pdf`, etc.

---

## Deploy to Railway (Recommended)

Railway is the easiest — free tier, no credit card required to start.

### Step 1: Push to GitHub

Create a new repo and push this code:

```bash
git init
git add .
git commit -m "PDF Chunker"
git remote add origin https://github.com/YOUR_USERNAME/pdf-chunker.git
git push -u origin main
```

### Step 2: Deploy on Railway

1. Go to [railway.app](https://railway.app)
2. Click **"Start a New Project"**
3. Select **"Deploy from GitHub Repo"**
4. Connect your GitHub account if needed
5. Select your `pdf-chunker` repo
6. Railway auto-detects the Dockerfile and deploys

### Step 3: Get Your URL

Once deployed (takes ~2 minutes):
1. Click on your service
2. Go to **Settings** → **Networking**
3. Click **"Generate Domain"**
4. You'll get a URL like `pdf-chunker-production-abc123.up.railway.app`

That's it. The chunker is live.

---

## Local Development

### Install Tesseract

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

### Run locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## Integration with QA Platform

The QA Platform UI will:

1. **POST** mega-PDF to `https://your-chunker.up.railway.app/split`
2. Receive ZIP with batch PDFs + manifest
3. Upload each batch PDF to Vertesia via `/objects` API
4. Fire parallel orchestrator calls with batch document IDs
5. Poll for results

Example manifest.json:
```json
{
  "totalPages": 800,
  "totalCustomers": 200,
  "totalBatches": 17,
  "batchSize": 12,
  "batches": [
    {
      "batchNumber": 1,
      "customerCount": 12,
      "pageStart": 1,
      "pageEnd": 48,
      "pageCount": 48,
      "filename": "batch_001.pdf",
      "customers": [
        {"name": "JOHN SMITH", "pageStart": 1, "pageEnd": 4, ...},
        ...
      ]
    },
    ...
  ]
}
```

---

## Notes

- OCR runs at 200 DPI — about 1-2 seconds per page
- 800-page PDF takes ~15-20 minutes to process
- Railway free tier: 500 hours/month, 512MB RAM (enough for this)
- Timeout set to 300 seconds for large PDFs

