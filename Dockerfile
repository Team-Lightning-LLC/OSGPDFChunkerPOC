"""
PDF Chunker - Customer Boundary Detection with OCR
Run: python app.py
Then open http://localhost:5000
"""

import os
import re
import json
import tempfile
import uuid
import threading
import shutil
import time
import zipfile
from flask import Flask, request, jsonify, render_template_string, send_file, send_file
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Job storage (in production, use Redis or a database)
jobs = {}
jobs_lock = threading.Lock()

class Job:
    def __init__(self, job_id, filename, batch_size):
        self.job_id = job_id
        self.filename = filename
        self.batch_size = batch_size
        self.status = "pending"  # pending, processing, done, error
        self.progress = 0
        self.total_pages = 0
        self.pages_complete = 0
        self.customers_found = 0
        self.error = None
        self.result_path = None
        self.manifest = None
        self.created_at = time.time()
    
    def to_dict(self):
        return {
            "jobId": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "totalPages": self.total_pages,
            "pagesComplete": self.pages_complete,
            "customersFound": self.customers_found,
            "error": self.error,
            "manifest": self.manifest if self.status == "done" else None,
        }

# US States for validation
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC','PR','VI','GU','AS','MP'
}

# Patterns
CITY_STATE_ZIP_RE = re.compile(r'^(.+?),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$')
STREET_RE = re.compile(r'^\d+\s+.{3,}')

# Common OCR noise prefixes from barcodes/QR codes to strip
NOISE_PREFIXES = re.compile(r'^[^A-Za-z0-9]*|^(fee|ate|Ee|EE|ae|ie|oe|\.,\s*_|\.\s*_|&\s*|[|!li]{1,3}\s*)+', re.I)

NON_NAME_PATTERNS = [
    re.compile(r'^P\.?\s*O\.?\s*BOX', re.I),
    re.compile(r'^SUITE', re.I),
    re.compile(r'^APT', re.I),
    re.compile(r'^DEPT', re.I),
    re.compile(r'^ATTN', re.I),
    re.compile(r'^RE:', re.I),
    re.compile(r'^RETURN\s+SERVICE', re.I),
    re.compile(r'^FIRST.?CLASS', re.I),
    re.compile(r'^PRESORTED', re.I),
    re.compile(r'^U\.?S\.?\s*POSTAGE', re.I),
    re.compile(r'^\d+\s*$'),  # Just numbers
    re.compile(r'^[A-Z]{2}\s+\d', re.I),  # State + number (barcode lines)
    re.compile(r'^[\d\s\-\+]+$'),  # Phone numbers, barcodes
    re.compile(r'Corporate Drive', re.I),  # Return address
    re.compile(r'Lake Zurich', re.I),  # Return address
]


def is_likely_person_name(text):
    """Check if text looks like a person's name"""
    if not text or len(text) < 3 or len(text) > 80:
        return False
    
    # Skip known non-name patterns
    for pattern in NON_NAME_PATTERNS:
        if pattern.search(text):
            return False
    
    # Should have at least 2 words
    words = text.split()
    if len(words) < 2:
        return False
    
    # Should be mostly alphabetic
    alpha_chars = sum(1 for c in text if c.isalpha() or c in " '-")
    if alpha_chars / len(text) < 0.8:
        return False
    
    return True


def clean_line(text):
    """Remove common OCR noise prefixes from barcode/QR code artifacts"""
    if not text:
        return text
    # Strip noise prefixes
    cleaned = NOISE_PREFIXES.sub('', text).strip()
    # Also strip leading punctuation and spaces
    cleaned = re.sub(r'^[^A-Za-z0-9]+', '', cleaned).strip()
    return cleaned if cleaned else text


def ocr_page(page, dpi=200):
    """OCR a single PDF page, return list of text lines"""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    
    # Get OCR with bounding boxes for position info
    text = pytesseract.image_to_string(img)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    return lines


def detect_customer_on_page(lines, page_num, zone_top_pct=0, zone_bottom_pct=40):
    """
    Detect customer address block in the top portion of a page.
    Returns customer dict or None.
    """
    # Only look at top portion of page (roughly first 40% of lines)
    zone_end = max(1, int(len(lines) * zone_bottom_pct / 100))
    zone_lines = lines[:zone_end]
    
    # Look for City, ST ZIP pattern
    for i, line in enumerate(zone_lines):
        line_raw = line.strip()
        line_clean = clean_line(line_raw)
        match = CITY_STATE_ZIP_RE.match(line_clean)
        
        if not match:
            continue
        
        city = match.group(1).strip()
        state = match.group(2).upper()
        zip_code = match.group(3)
        
        # Validate state
        if state not in US_STATES:
            continue
        
        # Skip if this looks like the return address (Lake Zurich, IL pattern)
        if 'ZURICH' in city.upper() or 'GREENVILLE' in city.upper():
            continue
        
        # Look backwards for street and name
        street = None
        name = None
        confidence = 'weak'
        address_lines = [line_clean]
        
        # Check line above for street address
        if i >= 1:
            prev_line_raw = zone_lines[i - 1].strip()
            prev_line = clean_line(prev_line_raw)
            if STREET_RE.match(prev_line):
                street = prev_line
                address_lines.insert(0, street)
                
                # Check line above street for name
                if i >= 2:
                    name_line_raw = zone_lines[i - 2].strip()
                    name_line = clean_line(name_line_raw)
                    if is_likely_person_name(name_line):
                        name = name_line
                        address_lines.insert(0, name)
                        confidence = 'strong'
                    # Sometimes there's a second name line (co-borrower)
                    elif i >= 3:
                        name_line_raw = zone_lines[i - 3].strip()
                        name_line = clean_line(name_line_raw)
                        if is_likely_person_name(name_line):
                            name = name_line
                            # Include co-borrower
                            co_name_raw = zone_lines[i - 2].strip()
                            co_name = clean_line(co_name_raw)
                            if is_likely_person_name(co_name):
                                name = f"{name} / {co_name}"
                            address_lines.insert(0, name)
                            confidence = 'strong'
                
                if not name:
                    confidence = 'medium'
            
            elif is_likely_person_name(prev_line):
                # Name directly above city line (no street)
                name = prev_line
                address_lines.insert(0, name)
                confidence = 'medium'
        
        return {
            'name': name or '(Name not detected)',
            'street': street or '(Street not detected)',
            'city': city,
            'state': state,
            'zip': zip_code,
            'cityStateZip': f"{city}, {state} {zip_code}",
            'pageStart': page_num,
            'pageEnd': None,
            'confidence': confidence,
            'addressLines': address_lines,
        }
    
    return None


def process_pdf(pdf_path, batch_size=12, job=None):
    """Process PDF and detect all customer boundaries - with parallel OCR"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    
    if job:
        job.total_pages = total_pages
        job.status = "processing"
    
    # Step 1: Render all pages to images (fast, sequential - PyMuPDF isn't thread-safe)
    page_images = []
    for page_num in range(total_pages):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        page_images.append((page_num, img_bytes))
        
        if job:
            job.pages_complete = page_num + 1
            job.progress = int(((page_num + 1) / total_pages) * 20)  # First 20% is rendering
    
    doc.close()
    
    # Step 2: OCR in parallel (configurable threads, default 6)
    NUM_WORKERS = int(os.environ.get('OCR_WORKERS', 6))
    page_results = [None] * total_pages
    
    def ocr_image(args):
        """OCR a single page image"""
        page_num, img_bytes = args
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        customer = detect_customer_on_page(lines, page_num + 1)
        return {
            'page_num': page_num,
            'lines': lines,
            'text': '\n'.join(lines),
            'customer': customer
        }
    
    completed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(ocr_image, args): args[0] for args in page_images}
        
        for future in as_completed(futures):
            result = future.result()
            page_results[result['page_num']] = result
            
            completed += 1
            if job:
                job.pages_complete = completed
                job.progress = 20 + int((completed / total_pages) * 75)  # 20-95%
    
    # Step 3: Process results in order
    customers = []
    page_texts = []
    
    for page_num in range(total_pages):
        result = page_results[page_num]
        page_texts.append({
            'pageNum': page_num + 1,
            'lines': result['lines'],
            'text': result['text']
        })
        
        if result['customer']:
            customer = result['customer']
            key = f"{customer['name']}|{customer['state']}|{customer['zip']}"
            existing_keys = [f"{c['name']}|{c['state']}|{c['zip']}" for c in customers]
            if key not in existing_keys:
                customers.append(customer)
    
    if job:
        job.customers_found = len(customers)
    
    # Assign page ranges
    for i, customer in enumerate(customers):
        if i < len(customers) - 1:
            customer['pageEnd'] = customers[i + 1]['pageStart'] - 1
        else:
            customer['pageEnd'] = total_pages
        
        customer['pageCount'] = customer['pageEnd'] - customer['pageStart'] + 1
    
    # Group into batches
    batches = []
    for i in range(0, len(customers), batch_size):
        batch_customers = customers[i:i + batch_size]
        
        # Calculate batch page range
        batch_page_start = batch_customers[0]['pageStart']
        batch_page_end = batch_customers[-1]['pageEnd']
        
        batches.append({
            'batchNumber': len(batches) + 1,
            'customerCount': len(batch_customers),
            'pageStart': batch_page_start,
            'pageEnd': batch_page_end,
            'pageCount': batch_page_end - batch_page_start + 1,
            'customers': [{**c, 'index': i + j + 1} for j, c in enumerate(batch_customers)]
        })
    
    # Build manifest
    manifest = {
        'totalPages': total_pages,
        'totalCustomers': len(customers),
        'totalBatches': len(batches),
        'batchSize': batch_size,
        'confidenceSummary': {
            'strong': sum(1 for c in customers if c['confidence'] == 'strong'),
            'medium': sum(1 for c in customers if c['confidence'] == 'medium'),
            'weak': sum(1 for c in customers if c['confidence'] == 'weak'),
        },
        'batches': batches,
    }
    
    return {
        'manifest': manifest,
        'customers': customers,
        'pageTexts': page_texts,
    }


def split_pdf_into_batches(pdf_path, batches, output_dir):
    """
    Split a PDF into batch files based on detected customer batches.
    Returns list of batch file paths.
    """
    doc = fitz.open(pdf_path)
    batch_files = []
    
    for batch in batches:
        batch_num = batch['batchNumber']
        page_start = batch['pageStart'] - 1  # Convert to 0-indexed
        page_end = batch['pageEnd']  # fitz.select uses exclusive end
        
        # Create new PDF with just this batch's pages
        batch_doc = fitz.open()
        batch_doc.insert_pdf(doc, from_page=page_start, to_page=page_end - 1)
        
        # Save batch file
        batch_filename = f"batch_{batch_num:03d}.pdf"
        batch_path = os.path.join(output_dir, batch_filename)
        batch_doc.save(batch_path)
        batch_doc.close()
        
        batch_files.append({
            'batchNumber': batch_num,
            'filename': batch_filename,
            'path': batch_path,
            'pageCount': batch['pageCount'],
            'customerCount': batch['customerCount'],
            'customers': batch['customers']
        })
    
    doc.close()
    return batch_files


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PDF Chunker — OCR Edition</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    :root {
      --bg: #f5f1ea;
      --surface: #fdfaf4;
      --text: #3d3830;
      --text-2: #6a6259;
      --text-3: #a09686;
      --border: #e5ddd0;
      --accent: #5a7a8a;
      --accent-soft: #eef2f4;
      --pass: #5a8a6a;
      --pass-soft: #f0f5ef;
      --fail: #c4785c;
      --fail-soft: #faf2ed;
      --warning: #b8943d;
      --warning-soft: #fdf8ec;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'IBM Plex Sans', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.5;
      font-size: 14px;
    }
    
    .header { max-width: 1100px; margin: 0 auto 24px; }
    .header h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
    .header p { font-size: 13px; color: var(--text-2); }
    
    .container { max-width: 1100px; margin: 0 auto; }
    
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 20px;
    }
    
    .panel-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-2);
      display: flex;
      justify-content: space-between;
    }
    
    .panel-body { padding: 16px; }
    
    .upload-zone {
      border: 1.5px dashed var(--border);
      border-radius: 10px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s ease;
    }
    
    .upload-zone:hover { border-color: var(--accent); background: var(--accent-soft); }
    .upload-zone p { font-size: 15px; margin-bottom: 4px; }
    .upload-zone small { font-size: 12px; color: var(--text-3); }
    
    .config-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 14px;
      padding: 10px 14px;
      background: var(--bg);
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    
    .config-row label { font-size: 12px; font-weight: 600; color: var(--text-2); }
    .config-row input[type="number"] {
      width: 60px; padding: 5px 8px;
      border: 1px solid var(--border); border-radius: 5px;
      font-family: inherit; font-size: 13px; text-align: center;
    }
    .config-row small { font-size: 11px; color: var(--text-3); }
    
    .btn {
      padding: 10px 20px;
      border-radius: 6px;
      font-family: inherit;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      border: none;
      transition: all 0.2s;
    }
    
    .btn-primary { background: var(--accent); color: white; }
    .btn-primary:hover { filter: brightness(1.1); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .processing {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      background: var(--accent-soft);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 16px;
    }
    
    .spinner {
      width: 20px; height: 20px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin { 100% { transform: rotate(360deg); } }
    
    .progress-container {
      margin-bottom: 16px;
    }
    
    .progress-bar-bg {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      height: 28px;
      overflow: hidden;
    }
    
    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--pass));
      transition: width 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 12px;
      font-weight: 600;
    }
    
    .progress-text {
      margin-top: 10px;
      font-size: 13px;
      color: var(--text-2);
    }
    
    .progress-detail {
      font-size: 12px;
      color: var(--text-3);
      margin-top: 4px;
    }
    
    .download-section {
      background: var(--pass-soft);
      border: 1px solid var(--pass);
      border-radius: 10px;
      padding: 20px;
      text-align: center;
      margin-bottom: 20px;
    }
    
    .download-section h3 {
      color: var(--pass);
      font-size: 16px;
      margin-bottom: 10px;
    }
    
    .download-section p {
      color: var(--text-2);
      margin-bottom: 14px;
    }
    
    .btn-download {
      background: var(--pass);
      color: white;
      padding: 12px 28px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      border: none;
    }
    
    .btn-download:hover {
      filter: brightness(1.1);
    }
    
    .stats-bar {
      display: flex;
      gap: 16px;
      padding: 10px 14px;
      background: var(--bg);
      border-radius: 8px;
      border: 1px solid var(--border);
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    
    .stat { display: flex; align-items: center; gap: 6px; font-size: 13px; }
    .stat strong { font-weight: 600; }
    .stat.pages { color: var(--accent); }
    .stat.customers { color: var(--pass); }
    .stat.batches { color: var(--warning); }
    
    .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 800px) { .results-grid { grid-template-columns: 1fr; } }
    
    .customer-list { max-height: 60vh; overflow-y: auto; }
    
    .batch-header {
      padding: 6px 12px;
      background: var(--text);
      color: var(--bg);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    
    .customer-item {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }
    
    .customer-num {
      width: 28px; height: 28px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 600;
      flex-shrink: 0;
    }
    
    .customer-num.strong { background: var(--pass-soft); color: var(--pass); }
    .customer-num.medium { background: var(--warning-soft); color: var(--warning); }
    .customer-num.weak { background: var(--fail-soft); color: var(--fail); }
    
    .customer-info { flex: 1; min-width: 0; }
    .customer-name { font-size: 13px; font-weight: 500; }
    .customer-address { font-size: 12px; color: var(--text-2); }
    .customer-meta { font-size: 11px; color: var(--text-3); margin-top: 2px; }
    
    .confidence {
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 600;
    }
    .confidence.strong { background: var(--pass-soft); color: var(--pass); }
    .confidence.medium { background: var(--warning-soft); color: var(--warning); }
    .confidence.weak { background: var(--fail-soft); color: var(--fail); }
    
    .page-badge {
      font-size: 11px;
      font-family: 'IBM Plex Mono', monospace;
      color: var(--text-3);
      white-space: nowrap;
    }
    
    .page-viewer { max-height: 60vh; overflow-y: auto; }
    
    .page-block {
      margin-bottom: 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }
    
    .page-block-header {
      padding: 6px 12px;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      font-size: 11px;
      font-weight: 600;
      color: var(--text-2);
      cursor: pointer;
      display: flex;
      justify-content: space-between;
    }
    
    .page-block-header:hover { background: var(--border); }
    .page-block-header .customer-tag { color: var(--pass); font-weight: 500; }
    
    .page-block-body {
      padding: 10px 12px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      white-space: pre-wrap;
      color: var(--text-2);
      max-height: 200px;
      overflow-y: auto;
      line-height: 1.6;
    }
    
    .page-block-body.collapsed { display: none; }
    
    .manifest-json {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      white-space: pre-wrap;
      background: var(--bg);
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      max-height: 60vh;
      overflow-y: auto;
    }
    
    .tab-bar { display: flex; border-bottom: 1px solid var(--border); }
    .tab-btn {
      flex: 1;
      padding: 10px;
      font-family: inherit;
      font-size: 12px;
      font-weight: 500;
      border: none;
      background: none;
      color: var(--text-3);
      cursor: pointer;
      position: relative;
    }
    .tab-btn.active { color: var(--accent); font-weight: 600; }
    .tab-btn.active::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 16px; right: 16px;
      height: 2px;
      background: var(--accent);
    }
    
    .tab-content { display: none; padding: 16px; }
    .tab-content.active { display: block; }
    
    .hidden { display: none !important; }
  </style>
</head>
<body>
  <div class="header">
    <h1>PDF Chunker — OCR Edition</h1>
    <p>Upload a multi-customer After Sample PDF. Uses OCR to detect customer boundaries even in vector-rendered PDFs.</p>
  </div>
  
  <div class="container">
    <!-- Upload Panel -->
    <div class="panel" id="uploadPanel">
      <div class="panel-header"><span>Upload</span></div>
      <div class="panel-body">
        <form id="uploadForm" enctype="multipart/form-data">
          <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" name="file" accept=".pdf" style="display:none">
            <p>Drop mega-PDF here or click to select</p>
            <small>Multi-customer After Sample document</small>
          </div>
          
          <div class="config-row">
            <label>Batch size:</label>
            <input type="number" id="batchSize" name="batch_size" value="12" min="1" max="50">
            <small>Customers per orchestrator batch (10-15 recommended)</small>
          </div>
          
          <div style="margin-top: 16px; text-align: center;">
            <button type="submit" class="btn btn-primary" id="processBtn" disabled>Process PDF</button>
          </div>
        </form>
        
        <div id="selectedFile" class="config-row hidden" style="margin-top: 12px;">
          <span id="fileName"></span>
        </div>
      </div>
    </div>
    
    <!-- Processing Panel -->
    <div class="panel hidden" id="processingPanel">
      <div class="panel-header"><span>Processing</span><span id="processingJobId"></span></div>
      <div class="panel-body">
        <div class="progress-container">
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="progressBar" style="width: 0%">0%</div>
          </div>
          <div class="progress-text" id="progressText">Starting OCR...</div>
          <div class="progress-detail" id="progressDetail">Preparing document...</div>
        </div>
      </div>
    </div>
    
    <!-- Results Panel -->
    <div id="resultsPanel" class="hidden">
      <div class="download-section" id="downloadSection">
        <h3>✓ Processing Complete</h3>
        <p id="downloadSummary">Ready to download</p>
        <button class="btn-download" onclick="downloadResults()">Download Batch PDFs (ZIP)</button>
      </div>
      
      <div class="stats-bar" id="statsBar"></div>
      
      <div class="results-grid">
        <div class="panel">
          <div class="panel-header">
            <span>Detected Customers</span>
            <span id="customerCount">0</span>
          </div>
          <div class="customer-list" id="customerList"></div>
        </div>
        
        <div class="panel">
          <div class="panel-header"><span>Manifest</span></div>
          <div class="panel-body">
            <pre class="manifest-json" id="manifestJson"></pre>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <script>
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const uploadForm = document.getElementById('uploadForm');
    
    let currentJobId = null;
    let pollInterval = null;
    
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        document.getElementById('fileName').textContent = fileInput.files[0].name;
        document.getElementById('selectedFile').classList.remove('hidden');
        processBtn.disabled = false;
      }
    });
    
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      document.getElementById('uploadPanel').classList.add('hidden');
      document.getElementById('processingPanel').classList.remove('hidden');
      document.getElementById('progressBar').style.width = '0%';
      document.getElementById('progressBar').textContent = '0%';
      document.getElementById('progressText').textContent = 'Uploading file...';
      document.getElementById('progressDetail').textContent = 'Starting job...';
      
      const formData = new FormData(uploadForm);
      
      try {
        // Start async job
        const response = await fetch('/split', {
          method: 'POST',
          body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
          alert('Error: ' + data.error);
          resetToUpload();
          return;
        }
        
        currentJobId = data.jobId;
        document.getElementById('processingJobId').textContent = 'Job: ' + currentJobId;
        document.getElementById('progressText').textContent = 'Job started, processing...';
        
        // Start polling
        pollInterval = setInterval(pollJobStatus, 1500);
        
      } catch (err) {
        alert('Error starting job: ' + err.message);
        resetToUpload();
      }
    });
    
    async function pollJobStatus() {
      if (!currentJobId) return;
      
      try {
        const response = await fetch('/status/' + currentJobId);
        const data = await response.json();
        
        if (data.error && !data.status) {
          clearInterval(pollInterval);
          alert('Error: ' + data.error);
          resetToUpload();
          return;
        }
        
        // Update progress bar
        const progress = data.progress || 0;
        document.getElementById('progressBar').style.width = progress + '%';
        document.getElementById('progressBar').textContent = progress + '%';
        
        // Update text
        if (data.totalPages > 0) {
          document.getElementById('progressText').textContent = 
            `Processing page ${data.pagesComplete || 0} of ${data.totalPages}`;
          document.getElementById('progressDetail').textContent = 
            `${data.customersFound || 0} customers detected`;
        }
        
        // Check if done
        if (data.status === 'done') {
          clearInterval(pollInterval);
          renderResults(data.manifest);
          document.getElementById('processingPanel').classList.add('hidden');
          document.getElementById('resultsPanel').classList.remove('hidden');
        } else if (data.status === 'error') {
          clearInterval(pollInterval);
          alert('Processing error: ' + (data.error || 'Unknown error'));
          resetToUpload();
        }
        
      } catch (err) {
        console.error('Poll error:', err);
        // Don't stop polling on network hiccups
      }
    }
    
    function resetToUpload() {
      document.getElementById('processingPanel').classList.add('hidden');
      document.getElementById('resultsPanel').classList.add('hidden');
      document.getElementById('uploadPanel').classList.remove('hidden');
      currentJobId = null;
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }
    
    function downloadResults() {
      if (currentJobId) {
        window.location.href = '/download/' + currentJobId;
      }
    }
    
    function renderResults(manifest) {
      // Download summary
      document.getElementById('downloadSummary').textContent = 
        `${manifest.totalCustomers} customers split into ${manifest.totalBatches} batch PDFs`;
      
      // Stats bar
      document.getElementById('statsBar').innerHTML = `
        <div class="stat pages"><strong>${manifest.totalPages}</strong> pages</div>
        <div class="stat customers"><strong>${manifest.totalCustomers}</strong> customers</div>
        <div class="stat batches"><strong>${manifest.totalBatches}</strong> batches of ${manifest.batchSize}</div>
        ${manifest.confidenceSummary.strong ? `<div class="stat"><strong>${manifest.confidenceSummary.strong}</strong> strong</div>` : ''}
        ${manifest.confidenceSummary.medium ? `<div class="stat"><strong>${manifest.confidenceSummary.medium}</strong> medium</div>` : ''}
        ${manifest.confidenceSummary.weak ? `<div class="stat"><strong>${manifest.confidenceSummary.weak}</strong> weak</div>` : ''}
      `;
      
      document.getElementById('customerCount').textContent = manifest.totalCustomers;
      
      // Customer list
      const list = document.getElementById('customerList');
      list.innerHTML = '';
      
      for (const batch of manifest.batches) {
        const batchHeader = document.createElement('div');
        batchHeader.className = 'batch-header';
        batchHeader.innerHTML = `<span>Batch ${batch.batchNumber} (${batch.filename})</span><span>${batch.customerCount} customers · ${batch.pageCount} pages</span>`;
        list.appendChild(batchHeader);
        
        for (const c of batch.customers) {
          const pageRange = c.pageStart === c.pageEnd ? `p.${c.pageStart}` : `p.${c.pageStart}–${c.pageEnd}`;
          
          const el = document.createElement('div');
          el.className = 'customer-item';
          el.innerHTML = `
            <div class="customer-num ${c.confidence}">${c.index}</div>
            <div class="customer-info">
              <div class="customer-name">${escapeHtml(c.name)}</div>
              <div class="customer-address">${escapeHtml(c.street)} · ${escapeHtml(c.cityStateZip)}</div>
              <div class="customer-meta">
                <span class="confidence ${c.confidence}">${c.confidence.toUpperCase()}</span>
                ${c.state} · ${c.pageCount} pg
              </div>
            </div>
            <div class="page-badge">${pageRange}</div>
          `;
          list.appendChild(el);
        }
      }
      
      // Manifest JSON
      document.getElementById('manifestJson').textContent = JSON.stringify(manifest, null, 2);
    }
    
    function escapeHtml(str) {
      return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    batch_size = int(request.form.get('batch_size', 12))
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        result = process_pdf(tmp_path, batch_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


def process_job_background(job, pdf_path, output_dir):
    """Background worker to process a PDF job"""
    try:
        # Process and detect boundaries
        result = process_pdf(pdf_path, job.batch_size, job)
        
        # Split into batch files
        batch_files = split_pdf_into_batches(
            pdf_path, 
            result['manifest']['batches'], 
            output_dir
        )
        
        # Create manifest file
        manifest_with_files = {
            **result['manifest'],
            'batches': batch_files
        }
        manifest_path = os.path.join(output_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest_with_files, f, indent=2)
        
        # Create zip file
        zip_path = os.path.join(output_dir, 'batches.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, 'manifest.json')
            for bf in batch_files:
                zf.write(bf['path'], bf['filename'])
        
        # Update job as complete
        with jobs_lock:
            job.status = "done"
            job.progress = 100
            job.result_path = zip_path
            job.manifest = manifest_with_files
        
    except Exception as e:
        with jobs_lock:
            job.status = "error"
            job.error = str(e)
    
    finally:
        # Clean up input PDF (but keep output dir for download)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


@app.route('/split', methods=['POST'])
def split():
    """
    Start async PDF processing job.
    Returns job ID immediately, process runs in background.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    batch_size = int(request.form.get('batch_size', 12))
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, file.filename, batch_size)
    
    # Create temp directories
    tmp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(tmp_dir, 'batches')
    os.makedirs(output_dir)
    
    # Save uploaded PDF
    pdf_path = os.path.join(tmp_dir, 'input.pdf')
    file.save(pdf_path)
    
    # Store job
    with jobs_lock:
        jobs[job_id] = job
    
    # Start background thread
    thread = threading.Thread(
        target=process_job_background,
        args=(job, pdf_path, output_dir),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        'jobId': job_id,
        'status': 'pending',
        'message': 'Job started. Poll /status/{jobId} for progress.'
    })


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    """Check job status and progress"""
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job.to_dict())


@app.route('/download/<job_id>', methods=['GET'])
def download(job_id):
    """Download completed job result as ZIP"""
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.status != 'done':
        return jsonify({'error': 'Job not complete', 'status': job.status}), 400
    
    if not job.result_path or not os.path.exists(job.result_path):
        return jsonify({'error': 'Result file not found'}), 404
    
    return send_file(
        job.result_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"batches_{job.filename.replace('.pdf', '')}.zip"
    )


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs (for debugging)"""
    with jobs_lock:
        return jsonify({
            'jobs': [job.to_dict() for job in jobs.values()]
        })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"Starting PDF Chunker on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
