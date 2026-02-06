"""
PDF Chunker - Customer Boundary Detection
Strategy: Try text extraction first (~1 sec), fall back to OCR if needed (~2-5 min).
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
from flask import Flask, request, jsonify, render_template_string, send_file
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# ---------------------------------------------------------------------------
# Job storage (in production, use Redis or a database)
# ---------------------------------------------------------------------------
jobs = {}
jobs_lock = threading.Lock()


class Job:
    def __init__(self, job_id, filename, batch_size):
        self.job_id = job_id
        self.filename = filename
        self.batch_size = batch_size
        self.status = "pending"
        self.phase = "queued"
        self.progress = 0
        self.total_pages = 0
        self.pages_complete = 0
        self.customers_found = 0
        self.error = None
        self.result_path = None
        self.manifest = None
        self.created_at = time.time()
        self.mode_used = None  # 'text' or 'ocr'

    def to_dict(self):
        return {
            "jobId": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "totalPages": self.total_pages,
            "pagesComplete": self.pages_complete,
            "customersFound": self.customers_found,
            "error": self.error,
            "modeUsed": self.mode_used,
            "manifest": self.manifest if self.status == "done" else None,
        }


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC','PR','VI','GU','AS','MP'
}

CITY_STATE_ZIP_RE = re.compile(r'^(.+?),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$')
STREET_RE = re.compile(r'^\d+\s+.{3,}')

NOISE_PREFIXES = re.compile(
    r'^[^A-Za-z0-9]*|^(fee|ate|Ee|EE|ae|ie|oe|\.,\s*_|\.\s*_|&\s*|[|!li]{1,3}\s*)+',
    re.I,
)

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
    re.compile(r'^\d+\s*$'),
    re.compile(r'^[A-Z]{2}\s+\d', re.I),
    re.compile(r'^[\d\s\-\+]+$'),
    re.compile(r'Corporate Drive', re.I),
    re.compile(r'Lake Zurich', re.I),
]

NUM_WORKERS = int(os.environ.get('OCR_WORKERS', 8))


def is_likely_person_name(text):
    if not text or len(text) < 3 or len(text) > 80:
        return False
    for pattern in NON_NAME_PATTERNS:
        if pattern.search(text):
            return False
    words = text.split()
    if len(words) < 2:
        return False
    alpha_chars = sum(1 for c in text if c.isalpha() or c in " '-")
    if alpha_chars / len(text) < 0.8:
        return False
    return True


def clean_line(text):
    if not text:
        return text
    cleaned = NOISE_PREFIXES.sub('', text).strip()
    cleaned = re.sub(r'^[^A-Za-z0-9]+', '', cleaned).strip()
    return cleaned if cleaned else text


# ---------------------------------------------------------------------------
# STRATEGY 1 — Text extraction (no OCR, ~1-2 seconds for 600 pages)
# ---------------------------------------------------------------------------

def detect_boundaries_text(pdf_path, job=None):
    """
    Try to detect customer boundaries using embedded text layer.
    Returns list of customer dicts, or None if text extraction is not viable.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    if job:
        job.total_pages = total_pages
        job.phase = "text_scan"
        job.status = "processing"

    # Pass 1: find all pages with "NOTICE OF SERVICING TRANSFER"
    boundary_pages = []
    for i in range(total_pages):
        text = doc[i].get_text()
        if "NOTICE OF SERVICING TRANSFER" in text:
            boundary_pages.append(i)
        if job and i % 50 == 0:
            job.progress = int((i / total_pages) * 30)

    # If we found fewer than 2, text extraction isn't viable
    if len(boundary_pages) < 2:
        doc.close()
        return None

    if job:
        job.phase = "text_extract"
        job.progress = 35

    # Pass 2: extract customer info from each boundary
    customers = []
    for idx, pg_idx in enumerate(boundary_pages):
        text = doc[pg_idx].get_text()

        # Extract loan number from the NOTICE page
        loan_old = None
        loan_new = None
        for line in text.split('\n'):
            if 'Old Loan Number' in line:
                loan_old = line.split(':')[-1].strip()
            if 'New Loan Number' in line:
                loan_new = line.split(':')[-1].strip()

        # Extract customer name + address from subsequent pages
        # The payment coupon page contains "next payment" followed by the name/address block
        name = None
        street = None
        city_state_zip = None
        city = None
        state = None
        zip_code = None
        address_lines = []

        for offset in range(1, 4):
            if pg_idx + offset >= total_pages:
                break
            ntext = doc[pg_idx + offset].get_text()
            lines = [l.strip() for l in ntext.split('\n') if l.strip()]

            for j, line in enumerate(lines):
                if 'next payment' in line.lower():
                    # Lines after "next payment" are: NAME, [CO-BORROWER], STREET, CITY ST ZIP
                    remaining = lines[j + 1:]
                    parsed = _parse_address_block(remaining)
                    if parsed:
                        name = parsed['name']
                        street = parsed.get('street')
                        city = parsed.get('city')
                        state = parsed.get('state')
                        zip_code = parsed.get('zip')
                        city_state_zip = parsed.get('cityStateZip')
                        address_lines = parsed.get('addressLines', [])
                    break
            if name:
                break

        # Calculate page range
        if idx + 1 < len(boundary_pages):
            page_end = boundary_pages[idx + 1]  # exclusive (0-indexed)
        else:
            page_end = total_pages  # last customer goes to end

        customers.append({
            'name': name or '(Name not detected)',
            'street': street or '(Street not detected)',
            'city': city or '',
            'state': state or 'XX',
            'zip': zip_code or '00000',
            'cityStateZip': city_state_zip or '(Not detected)',
            'loanOld': loan_old,
            'loanNew': loan_new,
            'pageStart': pg_idx + 1,      # 1-indexed for display
            'pageEnd': page_end,           # 1-indexed inclusive
            'pageCount': page_end - pg_idx,
            'confidence': 'strong' if name and street else 'medium' if name else 'weak',
            'addressLines': address_lines,
            'index': idx + 1,
        })

        if job:
            job.customers_found = len(customers)
            job.progress = 35 + int((idx / len(boundary_pages)) * 55)

    doc.close()
    return customers


def _parse_address_block(lines):
    """Parse a name/street/city-state-zip block from a list of text lines."""
    if not lines:
        return None

    # Walk lines looking for CITY, ST ZIP — that anchors everything
    for i, line in enumerate(lines):
        match = CITY_STATE_ZIP_RE.match(line)
        if match and match.group(2) in US_STATES:
            city = match.group(1).strip()
            state = match.group(2)
            zip_code = match.group(3)

            # Skip return-address cities
            if 'ZURICH' in city.upper() or 'GREENVILLE' in city.upper():
                continue

            result = {
                'city': city,
                'state': state,
                'zip': zip_code,
                'cityStateZip': f"{city}, {state} {zip_code}",
                'addressLines': [],
            }

            # Line above CSZ = street
            if i >= 1 and STREET_RE.match(lines[i - 1]):
                result['street'] = lines[i - 1]
                # Line(s) above street = name(s)
                if i >= 2 and is_likely_person_name(lines[i - 2]):
                    # Check for co-borrower
                    if i >= 3 and is_likely_person_name(lines[i - 3]):
                        result['name'] = f"{lines[i - 3]} / {lines[i - 2]}"
                    else:
                        result['name'] = lines[i - 2]
                elif i >= 2:
                    # Maybe only 1 line above street = name even if it doesn't pass heuristic
                    result['name'] = lines[i - 2]
            elif i >= 1 and is_likely_person_name(lines[i - 1]):
                result['name'] = lines[i - 1]

            if result.get('name'):
                result['addressLines'] = [
                    l for l in [result.get('name'), result.get('street'), result['cityStateZip']]
                    if l
                ]
                return result

    return None


# ---------------------------------------------------------------------------
# STRATEGY 2 — OCR fallback (parallel, adaptive, ~2-5 min for 600 pages)
# ---------------------------------------------------------------------------

def detect_boundaries_ocr(pdf_path, job=None):
    """
    OCR-based boundary detection.  Scans every page in parallel but only the
    top 40% of each page at 150 DPI for speed.  No interval extrapolation —
    we find every real boundary.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    if job:
        job.total_pages = total_pages
        job.status = "processing"
        job.phase = "rendering"

    # Step 1: Render top 40% of every page (fast — ~15 sec for 600 pages)
    page_images = {}
    for page_num in range(total_pages):
        page = doc[page_num]
        rect = page.rect
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + (rect.height * 0.4))
        pix = page.get_pixmap(dpi=150, clip=clip)
        page_images[page_num] = pix.tobytes("png")
        if job and page_num % 20 == 0:
            job.pages_complete = page_num + 1
            job.progress = int((page_num / total_pages) * 15)

    doc.close()

    if job:
        job.phase = "ocr"

    # Step 2: OCR all pages in parallel
    customers = []
    customers_lock = threading.Lock()
    completed = [0]

    def ocr_single(page_num):
        img = Image.open(io.BytesIO(page_images[page_num]))
        text = pytesseract.image_to_string(img, config='--psm 6')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        customer = _detect_customer_on_page_ocr(lines, page_num + 1)

        if customer:
            with customers_lock:
                # De-duplicate by name+state+zip
                key = f"{customer['name']}|{customer['state']}|{customer['zip']}"
                if not any(
                    f"{c['name']}|{c['state']}|{c['zip']}" == key for c in customers
                ):
                    customers.append(customer)

        completed[0] += 1
        if job and completed[0] % 10 == 0:
            job.pages_complete = completed[0]
            job.customers_found = len(customers)
            job.progress = 15 + int((completed[0] / total_pages) * 75)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(ocr_single, p) for p in range(total_pages)]
        for f in as_completed(futures):
            f.result()  # raise any exceptions

    # Sort by page
    customers.sort(key=lambda c: c['pageStart'])

    # Assign page ranges (actual boundaries, not extrapolated)
    for i, c in enumerate(customers):
        if i + 1 < len(customers):
            c['pageEnd'] = customers[i + 1]['pageStart'] - 1
        else:
            c['pageEnd'] = total_pages
        c['pageCount'] = c['pageEnd'] - c['pageStart'] + 1
        c['index'] = i + 1

    return customers


def _detect_customer_on_page_ocr(lines, page_num):
    """Detect customer address block via OCR lines. Returns dict or None."""
    zone_end = max(1, int(len(lines) * 0.4))
    zone_lines = lines[:zone_end]

    for i, line in enumerate(zone_lines):
        line_clean = clean_line(line.strip())
        match = CITY_STATE_ZIP_RE.match(line_clean)
        if not match:
            continue

        city = match.group(1).strip()
        state = match.group(2).upper()
        zip_code = match.group(3)

        if state not in US_STATES:
            continue
        if 'ZURICH' in city.upper() or 'GREENVILLE' in city.upper():
            continue

        street = None
        name = None
        confidence = 'weak'
        address_lines = [line_clean]

        if i >= 1:
            prev = clean_line(zone_lines[i - 1].strip())
            if STREET_RE.match(prev):
                street = prev
                address_lines.insert(0, street)
                if i >= 2:
                    name_line = clean_line(zone_lines[i - 2].strip())
                    if is_likely_person_name(name_line):
                        name = name_line
                        confidence = 'strong'
                        # Check for co-borrower
                        if i >= 3:
                            co = clean_line(zone_lines[i - 3].strip())
                            if is_likely_person_name(co):
                                name = f"{co} / {name_line}"
                        address_lines.insert(0, name)
                    elif i >= 3:
                        name_line = clean_line(zone_lines[i - 3].strip())
                        if is_likely_person_name(name_line):
                            name = name_line
                            confidence = 'strong'
                            address_lines.insert(0, name)
                if not name:
                    confidence = 'medium'
            elif is_likely_person_name(prev):
                name = prev
                confidence = 'medium'
                address_lines.insert(0, name)

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


# ---------------------------------------------------------------------------
# Orchestrator — tries text first, falls back to OCR
# ---------------------------------------------------------------------------

def process_pdf(pdf_path, batch_size=12, job=None):
    """
    Main entry point.
    1. Try text extraction (~1-2 sec)
    2. If that fails, fall back to full OCR (~2-5 min)
    3. Build batches and manifest
    """
    start = time.time()

    # --- Attempt 1: text extraction ---
    if job:
        job.phase = "text_scan"
        job.status = "processing"

    customers = detect_boundaries_text(pdf_path, job)

    if customers and len(customers) >= 2:
        mode = 'text'
        if job:
            job.mode_used = 'text'
    else:
        # --- Attempt 2: OCR fallback ---
        mode = 'ocr'
        if job:
            job.phase = "ocr_fallback"
            job.mode_used = 'ocr'
        customers = detect_boundaries_ocr(pdf_path, job)

    elapsed = time.time() - start

    if job:
        job.phase = "batching"
        job.progress = 92
        job.customers_found = len(customers)

    # Build batches
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    batches = []
    for i in range(0, len(customers), batch_size):
        batch_custs = customers[i : i + batch_size]
        batches.append({
            'batchNumber': len(batches) + 1,
            'customerCount': len(batch_custs),
            'pageStart': batch_custs[0]['pageStart'],
            'pageEnd': batch_custs[-1]['pageEnd'],
            'pageCount': batch_custs[-1]['pageEnd'] - batch_custs[0]['pageStart'] + 1,
            'customers': batch_custs,
        })

    # Page count distribution
    from collections import Counter
    page_counts = Counter(c['pageCount'] for c in customers)

    manifest = {
        'totalPages': total_pages,
        'totalCustomers': len(customers),
        'totalBatches': len(batches),
        'batchSize': batch_size,
        'mode': mode,
        'elapsedSeconds': round(elapsed, 2),
        'pageCountDistribution': dict(page_counts.most_common()),
        'confidenceSummary': {
            'strong': sum(1 for c in customers if c.get('confidence') == 'strong'),
            'medium': sum(1 for c in customers if c.get('confidence') == 'medium'),
            'weak': sum(1 for c in customers if c.get('confidence') == 'weak'),
        },
        'batches': batches,
    }

    return {'manifest': manifest, 'customers': customers}


# ---------------------------------------------------------------------------
# PDF splitting
# ---------------------------------------------------------------------------

def split_pdf_into_batches(pdf_path, batches, output_dir):
    doc = fitz.open(pdf_path)
    batch_files = []

    for batch in batches:
        batch_num = batch['batchNumber']
        page_start = batch['pageStart'] - 1   # 0-indexed
        page_end = batch['pageEnd'] - 1        # 0-indexed inclusive

        batch_doc = fitz.open()
        batch_doc.insert_pdf(doc, from_page=page_start, to_page=page_end)

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
            'customers': batch['customers'],
        })

    doc.close()
    return batch_files


# ---------------------------------------------------------------------------
# Background job worker
# ---------------------------------------------------------------------------

def process_job_background(job, pdf_path, output_dir):
    try:
        result = process_pdf(pdf_path, job.batch_size, job)

        # Split into batch PDFs
        batch_files = split_pdf_into_batches(
            pdf_path, result['manifest']['batches'], output_dir
        )

        manifest_with_files = {**result['manifest'], 'batches': batch_files}
        manifest_path = os.path.join(output_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest_with_files, f, indent=2, default=str)

        zip_path = os.path.join(output_dir, 'batches.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, 'manifest.json')
            for bf in batch_files:
                zf.write(bf['path'], bf['filename'])

        with jobs_lock:
            job.status = "done"
            job.progress = 100
            job.phase = "done"
            job.result_path = zip_path
            job.manifest = manifest_with_files

    except Exception as e:
        import traceback
        traceback.print_exc()
        with jobs_lock:
            job.status = "error"
            job.error = str(e)
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PDF Chunker</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
      --bg: #f5f1ea; --surface: #fdfaf4; --text: #3d3830; --text-2: #6a6259;
      --text-3: #a09686; --border: #e5ddd0; --accent: #5a7a8a; --accent-soft: #eef2f4;
      --pass: #5a8a6a; --pass-soft: #f0f5ef; --fail: #c4785c; --fail-soft: #faf2ed;
      --warning: #b8943d; --warning-soft: #fdf8ec;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'IBM Plex Sans', -apple-system, sans-serif;
      background: var(--bg); color: var(--text);
      min-height: 100vh; padding: 24px; line-height: 1.5; font-size: 14px;
    }

    .header { max-width: 1100px; margin: 0 auto 24px; }
    .header h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
    .header p { font-size: 13px; color: var(--text-2); }
    .container { max-width: 1100px; margin: 0 auto; }

    .panel {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; overflow: hidden; margin-bottom: 20px;
    }
    .panel-header {
      padding: 12px 16px; border-bottom: 1px solid var(--border);
      font-size: 12px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.04em; color: var(--text-2);
      display: flex; justify-content: space-between;
    }
    .panel-body { padding: 16px; }

    .upload-zone {
      border: 1.5px dashed var(--border); border-radius: 10px;
      padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.3s;
    }
    .upload-zone:hover { border-color: var(--accent); background: var(--accent-soft); }
    .upload-zone p { font-size: 15px; margin-bottom: 4px; }
    .upload-zone small { font-size: 12px; color: var(--text-3); }

    .config-row {
      display: flex; align-items: center; gap: 12px; margin-top: 14px;
      padding: 10px 14px; background: var(--bg); border-radius: 8px;
      border: 1px solid var(--border);
    }
    .config-row label { font-size: 12px; font-weight: 600; color: var(--text-2); }
    .config-row input[type="number"] {
      width: 60px; padding: 5px 8px; border: 1px solid var(--border);
      border-radius: 5px; font-family: inherit; font-size: 13px; text-align: center;
    }
    .config-row small { font-size: 11px; color: var(--text-3); }

    .btn {
      padding: 10px 20px; border-radius: 6px; font-family: inherit;
      font-size: 13px; font-weight: 500; cursor: pointer; border: none; transition: all 0.2s;
    }
    .btn-primary { background: var(--accent); color: white; }
    .btn-primary:hover { filter: brightness(1.1); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

    .progress-bar-bg {
      background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; height: 28px; overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%; background: linear-gradient(90deg, var(--accent), var(--pass));
      transition: width 0.3s; display: flex; align-items: center;
      justify-content: center; color: white; font-size: 12px; font-weight: 600;
    }
    .progress-text { margin-top: 10px; font-size: 13px; color: var(--text-2); }
    .progress-detail { font-size: 12px; color: var(--text-3); margin-top: 4px; }

    .download-section {
      background: var(--pass-soft); border: 1px solid var(--pass);
      border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;
    }
    .download-section h3 { color: var(--pass); font-size: 16px; margin-bottom: 10px; }
    .download-section p { color: var(--text-2); margin-bottom: 14px; }
    .btn-download {
      background: var(--pass); color: white; padding: 12px 28px;
      border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; border: none;
    }
    .btn-download:hover { filter: brightness(1.1); }

    .mode-badge {
      display: inline-block; padding: 3px 10px; border-radius: 4px;
      font-size: 11px; font-weight: 600; margin-left: 8px;
    }
    .mode-badge.text { background: var(--pass-soft); color: var(--pass); }
    .mode-badge.ocr { background: var(--warning-soft); color: var(--warning); }

    .stats-bar {
      display: flex; gap: 16px; padding: 10px 14px; background: var(--bg);
      border-radius: 8px; border: 1px solid var(--border); margin-bottom: 14px; flex-wrap: wrap;
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
      padding: 6px 12px; background: var(--text); color: var(--bg);
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.05em; display: flex; justify-content: space-between;
      position: sticky; top: 0; z-index: 5;
    }
    .customer-item {
      display: flex; align-items: flex-start; gap: 10px;
      padding: 10px 12px; border-bottom: 1px solid var(--border);
    }
    .customer-num {
      width: 28px; height: 28px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 600; flex-shrink: 0;
    }
    .customer-num.strong { background: var(--pass-soft); color: var(--pass); }
    .customer-num.medium { background: var(--warning-soft); color: var(--warning); }
    .customer-num.weak { background: var(--fail-soft); color: var(--fail); }
    .customer-info { flex: 1; min-width: 0; }
    .customer-name { font-size: 13px; font-weight: 500; }
    .customer-address { font-size: 12px; color: var(--text-2); }
    .customer-meta { font-size: 11px; color: var(--text-3); margin-top: 2px; }
    .confidence { padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }
    .confidence.strong { background: var(--pass-soft); color: var(--pass); }
    .confidence.medium { background: var(--warning-soft); color: var(--warning); }
    .confidence.weak { background: var(--fail-soft); color: var(--fail); }
    .page-badge {
      font-size: 11px; font-family: 'IBM Plex Mono', monospace;
      color: var(--text-3); white-space: nowrap;
    }
    .manifest-json {
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      white-space: pre-wrap; background: var(--bg); padding: 12px;
      border-radius: 8px; border: 1px solid var(--border);
      max-height: 60vh; overflow-y: auto;
    }
    .hidden { display: none !important; }
  </style>
</head>
<body>
  <div class="header">
    <h1>PDF Chunker</h1>
    <p>Upload a multi-customer After Sample PDF. Tries text extraction first (~1 sec), falls back to OCR if needed.</p>
  </div>

  <div class="container">
    <div class="panel" id="uploadPanel">
      <div class="panel-header"><span>Upload</span></div>
      <div class="panel-body">
        <form id="uploadForm" enctype="multipart/form-data">
          <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" name="file" accept=".pdf" style="display:none">
            <p>Drop PDF here or click to select</p>
            <small>Multi-customer After Sample document</small>
          </div>
          <div class="config-row">
            <label>Batch size:</label>
            <input type="number" id="batchSize" name="batch_size" value="12" min="1" max="50">
            <small>Customers per batch (10–15 recommended)</small>
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

    <div class="panel hidden" id="processingPanel">
      <div class="panel-header"><span>Processing</span><span id="processingJobId"></span></div>
      <div class="panel-body">
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="progressBar" style="width: 0%">0%</div>
        </div>
        <div class="progress-text" id="progressText">Starting...</div>
        <div class="progress-detail" id="progressDetail"></div>
      </div>
    </div>

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
    let currentJobId = null, pollInterval = null;

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        document.getElementById('fileName').textContent = fileInput.files[0].name;
        document.getElementById('selectedFile').classList.remove('hidden');
        processBtn.disabled = false;
      }
    });

    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      show('processingPanel'); hide('uploadPanel');
      setProgress(0, 'Uploading file...', '');

      const formData = new FormData(uploadForm);
      try {
        const resp = await fetch('/split', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) { alert('Error: ' + data.error); resetToUpload(); return; }
        currentJobId = data.jobId;
        document.getElementById('processingJobId').textContent = 'Job: ' + currentJobId;
        pollInterval = setInterval(pollJobStatus, 800);
      } catch (err) { alert('Error: ' + err.message); resetToUpload(); }
    });

    async function pollJobStatus() {
      if (!currentJobId) return;
      try {
        const data = (await (await fetch('/status/' + currentJobId)).json());
        if (data.error && !data.status) { clearInterval(pollInterval); alert(data.error); resetToUpload(); return; }

        setProgress(data.progress || 0);
        const p = data.phase || '';
        if (p === 'text_scan')     setStatus('Scanning embedded text...', `Page ${data.pagesComplete||0} of ${data.totalPages}`);
        else if (p === 'text_extract') setStatus('Extracting customer info from text...', `${data.customersFound||0} customers found`);
        else if (p === 'ocr_fallback') setStatus('No text layer found — falling back to OCR...', 'This may take a few minutes');
        else if (p === 'rendering') setStatus('Rendering pages...', `${data.pagesComplete||0} of ${data.totalPages}`);
        else if (p === 'ocr')       setStatus(`OCR processing...`, `${data.pagesComplete||0} of ${data.totalPages} pages · ${data.customersFound||0} customers`);
        else if (p === 'batching')  setStatus('Building batches...', `${data.customersFound||0} customers`);
        else if (p === 'done')      setStatus('Complete!', '');

        if (data.status === 'done') {
          clearInterval(pollInterval);
          renderResults(data.manifest, data.modeUsed);
          hide('processingPanel'); show('resultsPanel');
        } else if (data.status === 'error') {
          clearInterval(pollInterval);
          alert('Error: ' + (data.error || 'Unknown'));
          resetToUpload();
        }
      } catch (e) { /* ignore transient */ }
    }

    function setProgress(pct, text, detail) {
      const bar = document.getElementById('progressBar');
      bar.style.width = pct + '%'; bar.textContent = pct + '%';
      if (text !== undefined) document.getElementById('progressText').textContent = text;
      if (detail !== undefined) document.getElementById('progressDetail').textContent = detail;
    }
    function setStatus(text, detail) {
      document.getElementById('progressText').textContent = text;
      document.getElementById('progressDetail').textContent = detail;
    }
    function show(id) { document.getElementById(id).classList.remove('hidden'); }
    function hide(id) { document.getElementById(id).classList.add('hidden'); }
    function resetToUpload() { hide('processingPanel'); hide('resultsPanel'); show('uploadPanel'); currentJobId=null; clearInterval(pollInterval); }
    function downloadResults() { if (currentJobId) window.location.href = '/download/' + currentJobId; }

    function renderResults(manifest, modeUsed) {
      const modeLabel = modeUsed === 'text'
        ? '<span class="mode-badge text">⚡ TEXT EXTRACT</span>'
        : '<span class="mode-badge ocr">🔍 OCR</span>';

      document.getElementById('downloadSummary').innerHTML =
        `${manifest.totalCustomers} customers → ${manifest.totalBatches} batch PDFs` +
        ` in ${manifest.elapsedSeconds}s ${modeLabel}`;

      document.getElementById('statsBar').innerHTML = `
        <div class="stat pages"><strong>${manifest.totalPages}</strong> pages</div>
        <div class="stat customers"><strong>${manifest.totalCustomers}</strong> customers</div>
        <div class="stat batches"><strong>${manifest.totalBatches}</strong> batches of ${manifest.batchSize}</div>
        <div class="stat"><strong>${manifest.elapsedSeconds}s</strong></div>
        ${manifest.confidenceSummary.strong ? `<div class="stat"><strong>${manifest.confidenceSummary.strong}</strong> strong</div>` : ''}
        ${manifest.confidenceSummary.medium ? `<div class="stat"><strong>${manifest.confidenceSummary.medium}</strong> medium</div>` : ''}
        ${manifest.confidenceSummary.weak   ? `<div class="stat"><strong>${manifest.confidenceSummary.weak}</strong> weak</div>` : ''}
      `;

      document.getElementById('customerCount').textContent = manifest.totalCustomers;
      const list = document.getElementById('customerList');
      list.innerHTML = '';

      for (const batch of manifest.batches) {
        const hdr = document.createElement('div'); hdr.className = 'batch-header';
        hdr.innerHTML = `<span>Batch ${batch.batchNumber} (${batch.filename})</span><span>${batch.customerCount} cust · ${batch.pageCount} pg</span>`;
        list.appendChild(hdr);

        for (const c of batch.customers) {
          const pr = c.pageStart === c.pageEnd ? `p.${c.pageStart}` : `p.${c.pageStart}–${c.pageEnd}`;
          const el = document.createElement('div'); el.className = 'customer-item';
          el.innerHTML = `
            <div class="customer-num ${c.confidence}">${c.index}</div>
            <div class="customer-info">
              <div class="customer-name">${esc(c.name)}</div>
              <div class="customer-address">${esc(c.street||'')} · ${esc(c.cityStateZip||'')}</div>
              <div class="customer-meta">
                <span class="confidence ${c.confidence}">${(c.confidence||'').toUpperCase()}</span>
                ${c.state} · ${c.pageCount} pg
                ${c.loanOld ? ' · Loan ' + c.loanOld : ''}
              </div>
            </div>
            <div class="page-badge">${pr}</div>`;
          list.appendChild(el);
        }
      }

      // Clean manifest for display (remove file paths)
      const displayManifest = JSON.parse(JSON.stringify(manifest));
      (displayManifest.batches||[]).forEach(b => { delete b.path; (b.customers||[]).forEach(c => delete c.addressLines); });
      document.getElementById('manifestJson').textContent = JSON.stringify(displayManifest, null, 2);
    }

    function esc(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  </script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/split', methods=['POST'])
def split():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    batch_size = int(request.form.get('batch_size', 12))

    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, file.filename, batch_size)

    tmp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(tmp_dir, 'batches')
    os.makedirs(output_dir)

    pdf_path = os.path.join(tmp_dir, 'input.pdf')
    file.save(pdf_path)

    with jobs_lock:
        jobs[job_id] = job

    thread = threading.Thread(
        target=process_job_background,
        args=(job, pdf_path, output_dir),
        daemon=True,
    )
    thread.start()

    return jsonify({'jobId': job_id, 'status': 'pending'})


@app.route('/status/<job_id>')
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job.to_dict())


@app.route('/download/<job_id>')
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.status != 'done':
        return jsonify({'error': 'Job not complete'}), 400
    if not job.result_path or not os.path.exists(job.result_path):
        return jsonify({'error': 'Result file not found'}), 404
    return send_file(
        job.result_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"batches_{job.filename.replace('.pdf', '')}.zip",
    )


@app.route('/jobs')
def list_jobs():
    with jobs_lock:
        return jsonify({'jobs': [j.to_dict() for j in jobs.values()]})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"Starting PDF Chunker on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
