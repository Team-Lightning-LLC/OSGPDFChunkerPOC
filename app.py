"""
PDF Chunker - Customer Boundary Detection + Vertesia Upload
Runs as a microservice. QA Platform sends PDFs here.

Endpoints:
  POST /split                    — Split only, returns jobId (poll + download batches)
  POST /split-and-upload         — Split + upload batches to Vertesia, returns doc IDs
  GET  /status/<jobId>           — Poll job progress (includes diagnostics on error)
  GET  /batch/<jobId>/<batchNum> — Download individual batch PDF
  GET  /download/<jobId>         — Download all batches as ZIP
  POST /diagnose                 — Upload PDF, get detection diagnostics (no split)
  GET  /health                   — Health check

Strategy: text extraction first (~2s for 600 pages), OCR fallback if needed.
Boundary detection: address-frequency classification. Corporate addresses repeat
across customer packets (multiple non-consecutive runs); customer addresses cluster
in a single consecutive run. No hardcoded strings or exclusions.
"""

import os, re, json, tempfile, uuid, threading, time, zipfile, io, shutil
from collections import Counter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import fitz  # PyMuPDF
import requests as http_requests  # renamed to avoid clash with flask.request

# Optional: OCR fallback
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

VERTESIA_API_BASE = os.environ.get('VERTESIA_API_BASE', 'https://api.vertesia.io/api/v1')

# ---------------------------------------------------------------------------
# Job storage (single-worker; use Redis if scaling)
# ---------------------------------------------------------------------------
jobs = {}
jobs_lock = threading.Lock()


class Job:
    def __init__(self, job_id, filename, batch_size):
        self.job_id = job_id
        self.filename = filename
        self.batch_size = batch_size
        self.status = "pending"          # pending | processing | done | error
        self.phase = "queued"            # queued | text_scan | text_extract |
                                         # ocr_fallback | rendering | ocr |
                                         # batching | uploading | done
        self.progress = 0                # 0-100
        self.total_pages = 0
        self.pages_complete = 0
        self.customers_found = 0
        self.error = None
        self.result_path = None          # ZIP path (for /split jobs)
        self.output_dir = None           # dir with batch PDFs
        self.manifest = None
        self.created_at = time.time()
        self.mode_used = None            # 'text' or 'ocr'
        self.diagnostics = None          # debug info for troubleshooting

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
            "diagnostics": self.diagnostics,
            "manifest": self.manifest if self.status == "done" else None,
        }


# ---------------------------------------------------------------------------
# Shared address-parsing helpers
# ---------------------------------------------------------------------------
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
    'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV',
    'NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
    'TX','UT','VT','VA','WA','WV','WI','WY','DC','PR','VI','GU','AS','MP',
}
CITY_STATE_ZIP_RE = re.compile(r'^(.+?),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$')
STREET_RE = re.compile(r'^\d+\s+.{3,}')
NOISE_PREFIXES = re.compile(
    r'^[^A-Za-z0-9]*|^(fee|ate|Ee|EE|ae|ie|oe|\.,\s*_|\.\s*_|&\s*|[|!li]{1,3}\s*)+',
    re.I,
)
NON_NAME_PATTERNS = [re.compile(p, re.I) for p in [
    r'^P\.?\s*O\.?\s*BOX', r'^SUITE', r'^APT', r'^DEPT', r'^ATTN', r'^RE:',
    r'^RETURN\s+SERVICE', r'^FIRST.?CLASS', r'^PRESORTED',
    r'^U\.?S\.?\s*POSTAGE', r'^\d+\s*$', r'^[A-Z]{2}\s+\d',
    r'^[\d\s\-\+]+$', r'(?i)corporate\s+drive', r'(?i)department',
    r'(?i)customer\s+service', r'(?i)mortgage', r'(?i)servic(?:er|ing)',
    r'(?i)payment\s+address', r'(?i)P\.?\s*O\.?\s*Box',
]]
LOAN_RE = re.compile(r'(?:Old|New|Loan)\s*(?:Loan\s*)?Number\s*[:\-]?\s*(\d[\d\-]+)', re.I)
NUM_WORKERS = int(os.environ.get('OCR_WORKERS', 8))


def is_likely_person_name(t):
    if not t or len(t) < 3 or len(t) > 80:
        return False
    if any(p.search(t) for p in NON_NAME_PATTERNS):
        return False
    if len(t.split()) < 2:
        return False
    return sum(1 for c in t if c.isalpha() or c in " '-") / len(t) >= 0.8


def clean_line(t):
    if not t:
        return t
    c = NOISE_PREFIXES.sub('', t).strip()
    c = re.sub(r'^[^A-Za-z0-9]+', '', c).strip()
    return c or t


def _extract_address_from_lines(lines, i):
    """Given lines and index i of a CSZ match, extract name + street above it."""
    name = street = None
    if i >= 1 and STREET_RE.match(lines[i - 1].strip()):
        street = lines[i - 1].strip()
        if i >= 2 and is_likely_person_name(lines[i - 2].strip()):
            name = lines[i - 2].strip()
            if i >= 3 and is_likely_person_name(lines[i - 3].strip()):
                name = f"{lines[i - 3].strip()} / {name}"
    elif i >= 1 and is_likely_person_name(lines[i - 1].strip()):
        name = lines[i - 1].strip()
    return name, street


def _extract_loans_from_text(text):
    """Pull Old/New Loan Numbers from page text."""
    lo = ln = None
    for line in text.split('\n'):
        if 'old' in line.lower() and 'loan' in line.lower():
            m = LOAN_RE.search(line)
            if m:
                lo = m.group(1)
        if 'new' in line.lower() and 'loan' in line.lower():
            m = LOAN_RE.search(line)
            if m:
                ln = m.group(1)
    return lo, ln


def _classify_corporate_addresses(csz_pages, total_pages):
    """
    Corporate addresses appear across a large fraction of the document
    (on nearly every customer packet). Customer addresses — even if shared
    by a few customers in the same city — appear on a small fraction.

    Heuristic: corporate if the address appears in non-consecutive runs
    AND covers more than 20% of total pages.
    """
    corp = set()
    for csz, pages in csz_pages.items():
        pages_sorted = sorted(set(pages))
        runs = 1
        for i in range(1, len(pages_sorted)):
            if pages_sorted[i] - pages_sorted[i - 1] > 1:
                runs += 1
        page_fraction = len(pages_sorted) / total_pages
        if runs > 1 and page_fraction > 0.2:
            corp.add(csz)
    return corp


# ---------------------------------------------------------------------------
# Text extraction (fast path — ~1-2 sec for 600 pages)
# ---------------------------------------------------------------------------
def detect_boundaries_text(pdf_path, job=None):
    doc = fitz.open(pdf_path)
    total = doc.page_count
    print(f"[DETECT-TEXT] Starting text scan: {total} pages")
    if job:
        job.total_pages = total
        job.phase = "text_scan"
        job.status = "processing"

    # --- Quality gate: is the text layer substantial enough to trust? ---
    MIN_TEXT_CHARS = 200
    page_text_lengths = []
    for p in range(total):
        text_len = len(doc[p].get_text().strip())
        page_text_lengths.append(text_len)

    pages_with_substance = sum(1 for l in page_text_lengths if l >= MIN_TEXT_CHARS)
    substance_ratio = pages_with_substance / total if total else 0
    print(f"[DETECT-TEXT] Text quality: {pages_with_substance}/{total} pages have >= {MIN_TEXT_CHARS} chars ({substance_ratio:.0%})")

    if substance_ratio < 0.5:
        print(f"[DETECT-TEXT] Text layer too thin ({substance_ratio:.0%} substantial pages). Falling back to OCR")
        doc.close()
        if job:
            job.diagnostics = {
                'stage': 'text_quality_gate',
                'reason': 'text_layer_too_thin',
                'total_pages': total,
                'pages_with_substantial_text': pages_with_substance,
                'substance_ratio': round(substance_ratio, 3),
                'threshold': '50% of pages must have >= 200 chars',
            }
        return None

    # --- Pass 1: extract all addresses from every page ---
    page_addr_details = {}  # page -> list of address dicts
    csz_pages = {}          # csz_key -> list of pages it appears on

    for p in range(total):
        text = doc[p].get_text()
        if not text.strip():
            page_addr_details[p] = []
            continue

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        addrs = []
        seen = set()
        for i, line in enumerate(lines):
            m = CITY_STATE_ZIP_RE.match(line)
            if not m or m.group(2) not in US_STATES:
                continue
            city, state, zc = m.group(1).strip(), m.group(2), m.group(3)
            key = f"{city.upper()}, {state} {zc}"
            if key in seen:
                continue
            seen.add(key)

            name, street = _extract_address_from_lines(lines, i)
            addrs.append({
                'csz_key': key, 'city': city, 'state': state, 'zip': zc,
                'name': name, 'street': street or '',
            })
            csz_pages.setdefault(key, []).append(p)

        page_addr_details[p] = addrs
        if job and p % 50 == 0:
            job.progress = int((p / total) * 30)

    # No addresses found at all → no text layer or unrecognizable format
    if not csz_pages:
        pages_with_text = sum(1 for p in range(total) if page_addr_details.get(p))
        print(f"[DETECT-TEXT] No addresses found. Pages with any text content: {pages_with_text}/{total}")
        print(f"[DETECT-TEXT] Falling back to OCR")
        doc.close()
        if job:
            job.diagnostics = {
                'stage': 'text_scan',
                'reason': 'no_addresses_found',
                'total_pages': total,
                'pages_with_any_content': pages_with_text,
            }
        return None

    # --- Pass 2: classify corporate vs customer addresses ---
    corp = _classify_corporate_addresses(csz_pages, total)
    print(f"[DETECT-TEXT] Addresses found: {len(csz_pages)} unique CSZs")
    for csz, pages in csz_pages.items():
        label = "CORP" if csz in corp else "CUST"
        print(f"[DETECT-TEXT]   [{label}] {csz}: {len(pages)} pages")
    print(f"[DETECT-TEXT] Corporate: {len(corp)}, Customer: {len(csz_pages) - len(corp)}")

    if job:
        job.phase = "text_extract"
        job.progress = 35

    # --- Pass 3: walk pages, find customer address per page ---
    # Text path: require name since text extraction gives clean names.
    # This filters out corporate addresses that lack person names
    # (e.g. "P.O. Box 10826" / "1 Corporate Drive").
    page_customer = [None] * total
    page_detail = [None] * total
    for p in range(total):
        for a in page_addr_details[p]:
            if a['csz_key'] not in corp and a['name']:
                page_customer[p] = a['csz_key']
                page_detail[p] = a
                break

    # --- Pass 4: boundaries where customer address changes ---
    boundaries = []
    current = None
    for p in range(total):
        csz = page_customer[p]
        if csz and csz != current:
            boundaries.append(p)
            current = csz

    if not boundaries:
        print(f"[DETECT-TEXT] No boundaries found after filtering. Falling back to OCR")
        doc.close()
        if job:
            job.diagnostics = {
                'stage': 'text_boundary_detect',
                'reason': 'all_addresses_classified_corporate',
                'all_addresses': {k: len(v) for k, v in csz_pages.items()},
                'corporate_addresses': list(corp),
                'surviving_addresses': [k for k in csz_pages if k not in corp],
            }
        return None

    # --- Sanity check: does the result make sense? ---
    avg_pages_per_customer = total / len(boundaries)
    if len(boundaries) == 1 and total > 10:
        print(f"[DETECT-TEXT] Only 1 boundary in {total}-page doc — likely bad detection. Falling back to OCR")
        doc.close()
        if job:
            job.diagnostics = {
                'stage': 'text_sanity_check',
                'reason': 'single_boundary_in_large_doc',
                'total_pages': total,
                'boundaries_found': 1,
                'avg_pages_per_customer': round(avg_pages_per_customer, 1),
            }
        return None

    print(f"[DETECT-TEXT] Boundaries: {len(boundaries)} at pages {[b+1 for b in boundaries]}")

    # Always populate diagnostics for visibility
    if job:
        addr_analysis = {}
        for k, v in csz_pages.items():
            pages_sorted = sorted(set(v))
            runs = 1
            for i in range(1, len(pages_sorted)):
                if pages_sorted[i] - pages_sorted[i-1] > 1:
                    runs += 1
            # Check if any occurrence had a name
            has_name = any(
                a['name'] for p in pages_sorted
                for a in page_addr_details.get(p, []) if a['csz_key'] == k
            )
            addr_analysis[k] = {
                'pages': pages_sorted[:20],
                'page_count': len(pages_sorted),
                'runs': runs,
                'fraction': round(len(pages_sorted) / total, 3),
                'is_corporate': k in corp,
                'has_person_name': has_name,
            }
        job.diagnostics = {
            'stage': 'text_boundary_detect',
            'total_pages': total,
            'address_analysis': addr_analysis,
            'corporate_addresses': list(corp),
            'surviving_addresses': [k for k in csz_pages if k not in corp],
            'boundaries_found': len(boundaries),
            'boundary_pages': [b + 1 for b in boundaries],
            'page_to_customer': {
                str(p + 1): page_customer[p] for p in range(total) if page_customer[p]
            },
        }

    # --- Build customer list ---
    customers = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else total - 1
        d = page_detail[start]

        # Extract loan numbers from boundary page and neighbors
        lo = ln = None
        for scan_p in range(start, min(start + 3, total)):
            plo, pln = _extract_loans_from_text(doc[scan_p].get_text())
            if plo and not lo:
                lo = plo
            if pln and not ln:
                ln = pln

        customers.append({
            'name': d['name'] or '(Not detected)',
            'street': d['street'],
            'city': d['city'],
            'state': d['state'],
            'zip': d['zip'],
            'cityStateZip': d['csz_key'],
            'loanOld': lo,
            'loanNew': ln,
            'pageStart': start + 1,
            'pageEnd': end + 1,
            'pageCount': end - start + 1,
            'confidence': 'strong' if d['name'] and d['street'] else 'medium' if d['name'] else 'weak',
            'addressLines': [l for l in [d['name'], d['street'], d['csz_key']] if l],
            'index': idx + 1,
        })
        if job:
            job.customers_found = len(customers)
            job.progress = 35 + int((idx / len(boundaries)) * 55)

    doc.close()
    return customers


# ---------------------------------------------------------------------------
# OCR fallback (if text extraction fails)
# ---------------------------------------------------------------------------
def detect_boundaries_ocr(pdf_path, job=None):
    if not HAS_OCR:
        raise RuntimeError("No text layer found and OCR (pytesseract) not installed")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    doc = fitz.open(pdf_path)
    total = doc.page_count
    print(f"[DETECT-OCR] Starting OCR scan: {total} pages, clip=60%, workers={NUM_WORKERS}")
    if job:
        job.total_pages = total
        job.status = "processing"
        job.phase = "rendering"

    imgs = {}
    for p in range(total):
        pg = doc[p]
        r = pg.rect
        clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + (r.height * 0.6))
        imgs[p] = pg.get_pixmap(dpi=150, clip=clip).tobytes("png")
        if job and p % 20 == 0:
            job.pages_complete = p + 1
            job.progress = int((p / total) * 15)
    doc.close()

    if job:
        job.phase = "ocr"

    # --- OCR all pages, collect raw address hits ---
    page_ocr_addrs = {}  # page -> list of address dicts
    csz_pages = {}       # csz -> list of pages
    ocr_lock = threading.Lock()
    done = [0]

    def ocr_page(p):
        img = Image.open(io.BytesIO(imgs[p]))
        text = pytesseract.image_to_string(img, config='--psm 6')
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        addrs = []
        seen = set()
        for i, line in enumerate(lines):
            lc = clean_line(line)
            m = CITY_STATE_ZIP_RE.match(lc)
            if not m or m.group(2).upper() not in US_STATES:
                continue
            city = m.group(1).strip()
            state = m.group(2).upper()
            zc = m.group(3)
            key = f"{city.upper()}, {state} {zc}"
            if key in seen:
                continue
            seen.add(key)

            name, street = _extract_address_from_lines(
                [clean_line(z) for z in lines], i,
            )
            addrs.append({
                'csz_key': key, 'city': city, 'state': state, 'zip': zc,
                'name': name, 'street': street or '',
            })

        with ocr_lock:
            page_ocr_addrs[p] = addrs
            for a in addrs:
                csz_pages.setdefault(a['csz_key'], []).append(p)

        done[0] += 1
        if job and done[0] % 10 == 0:
            job.pages_complete = done[0]
            job.progress = 15 + int((done[0] / total) * 55)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for f in as_completed([ex.submit(ocr_page, p) for p in range(total)]):
            f.result()

    # --- Classify corporate vs customer (same logic as text path) ---
    corp = _classify_corporate_addresses(csz_pages, total)

    print(f"[DETECT-OCR] OCR complete. Addresses found: {len(csz_pages)} unique CSZs")
    for csz, pages in csz_pages.items():
        label = "CORP" if csz in corp else "CUST"
        print(f"[DETECT-OCR]   [{label}] {csz}: {len(pages)} pages (pages: {sorted(pages)[:10]})")
    print(f"[DETECT-OCR] Corporate: {len(corp)}, Customer: {len(csz_pages) - len(corp)}")
    pages_with_any = sum(1 for p in range(total) if page_ocr_addrs.get(p))
    print(f"[DETECT-OCR] Pages with any address: {pages_with_any}/{total}")

    if job:
        job.phase = "text_extract"
        job.progress = 75
        # Capture OCR diagnostics regardless of outcome
        job.diagnostics = {
            'stage': 'ocr',
            'total_pages': total,
            'pages_with_addresses': sum(1 for p in range(total) if page_ocr_addrs.get(p)),
            'unique_addresses': {k: {'count': len(v), 'pages': sorted(v)[:10]} for k, v in csz_pages.items()},
            'corporate_addresses': list(corp),
            'surviving_addresses': [k for k in csz_pages if k not in corp],
        }

    # --- Walk pages in order, find boundaries ---
    # Use CSZ alone for boundary detection; name is metadata, not the signal
    page_customer = [None] * total
    page_detail = [None] * total
    for p in range(total):
        for a in page_ocr_addrs.get(p, []):
            if a['csz_key'] not in corp:
                page_customer[p] = a['csz_key']
                page_detail[p] = a
                break

    boundaries = []
    current = None
    for p in range(total):
        csz = page_customer[p]
        if csz and csz != current:
            boundaries.append(p)
            current = csz

    if job:
        job.diagnostics['boundaries_found'] = len(boundaries)
        job.diagnostics['boundary_pages'] = [b + 1 for b in boundaries]
        job.diagnostics['page_to_customer'] = {
            str(p + 1): page_customer[p] for p in range(total) if page_customer[p]
        }

    print(f"[DETECT-OCR] Boundaries: {len(boundaries)} at pages {[b+1 for b in boundaries]}")
    if not boundaries:
        print(f"[DETECT-OCR] *** NO BOUNDARIES FOUND — entire PDF will fail ***")

    # --- Build customer list ---
    customers = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else total - 1
        d = page_detail[start]
        customers.append({
            'name': d['name'] or '(Not detected)',
            'street': d['street'],
            'city': d['city'],
            'state': d['state'],
            'zip': d['zip'],
            'cityStateZip': d['csz_key'],
            'loanOld': None,
            'loanNew': None,
            'pageStart': start + 1,
            'pageEnd': end + 1,
            'pageCount': end - start + 1,
            'confidence': 'strong' if d['name'] and d['street'] else 'medium' if d['name'] else 'weak',
            'addressLines': [l for l in [d['name'], d['street'], d['csz_key']] if l],
            'index': idx + 1,
        })
        if job:
            job.customers_found = len(customers)
            job.progress = 75 + int((idx / max(1, len(boundaries))) * 15)

    return customers


# ---------------------------------------------------------------------------
# Main processor — text first, OCR fallback
# ---------------------------------------------------------------------------
def process_pdf(pdf_path, batch_size=12, job=None):
    start = time.time()
    if job:
        job.phase = "text_scan"
        job.status = "processing"

    custs = detect_boundaries_text(pdf_path, job)
    if custs:
        mode = 'text'
        if job:
            job.mode_used = 'text'
    else:
        mode = 'ocr'
        if job:
            job.phase = "ocr_fallback"
            job.mode_used = 'ocr'
        custs = detect_boundaries_ocr(pdf_path, job)

    if not custs:
        msg = (
            f"No customer boundaries detected in {job.filename if job else 'PDF'} "
            f"(mode={mode}). Check Railway logs for [DETECT-TEXT] / [DETECT-OCR] details."
        )
        print(f"[PROCESS] *** FAILED: {msg}")
        raise RuntimeError(msg)

    elapsed = time.time() - start
    print(f"[PROCESS] Done: {len(custs)} customers found via {mode} in {elapsed:.1f}s")
    if job:
        job.phase = "batching"
        job.progress = 92
        job.customers_found = len(custs)

    doc = fitz.open(pdf_path)
    total = doc.page_count
    doc.close()

    batches = []
    for i in range(0, len(custs), batch_size):
        bc = custs[i:i + batch_size]
        batches.append({
            'batchNumber': len(batches) + 1,
            'customerCount': len(bc),
            'pageStart': bc[0]['pageStart'],
            'pageEnd': bc[-1]['pageEnd'],
            'pageCount': bc[-1]['pageEnd'] - bc[0]['pageStart'] + 1,
            'customers': bc,
        })

    manifest = {
        'totalPages': total,
        'totalCustomers': len(custs),
        'totalBatches': len(batches),
        'batchSize': batch_size,
        'mode': mode,
        'elapsedSeconds': round(elapsed, 2),
        'pageCountDistribution': dict(Counter(c['pageCount'] for c in custs).most_common()),
        'confidenceSummary': {
            'strong': sum(1 for c in custs if c.get('confidence') == 'strong'),
            'medium': sum(1 for c in custs if c.get('confidence') == 'medium'),
            'weak': sum(1 for c in custs if c.get('confidence') == 'weak'),
        },
        'batches': batches,
    }
    return {'manifest': manifest, 'customers': custs}


# ---------------------------------------------------------------------------
# PDF splitting
# ---------------------------------------------------------------------------
def split_pdf_into_batches(pdf_path, batches, output_dir, naming_prefix=None):
    doc = fitz.open(pdf_path)
    files = []
    for b in batches:
        bd = fitz.open()
        bd.insert_pdf(doc, from_page=b['pageStart'] - 1, to_page=b['pageEnd'] - 1)
        if naming_prefix:
            fn = f"{naming_prefix}_Batch_{b['batchNumber']:03d}.pdf"
        else:
            fn = f"batch_{b['batchNumber']:03d}.pdf"
        fp = os.path.join(output_dir, fn)
        bd.save(fp)
        bd.close()
        files.append({**b, 'filename': fn, 'path': fp})
    doc.close()
    return files


# ---------------------------------------------------------------------------
# Vertesia upload helper
# ---------------------------------------------------------------------------
def upload_to_vertesia(file_path, filename, jwt_token, collection_id=None):
    """Upload a single PDF to Vertesia, return document ID."""
    headers = {'Authorization': f'Bearer {jwt_token}', 'Content-Type': 'application/json'}

    # 1. Get presigned upload URL
    resp = http_requests.post(
        f'{VERTESIA_API_BASE}/objects/upload-url',
        headers=headers,
        json={'name': filename, 'mime_type': 'application/pdf'},
    )
    if not resp.ok:
        raise Exception(f'Upload URL failed: {resp.status_code} {resp.text[:200]}')
    upload_data = resp.json()

    # 2. PUT file to presigned URL
    with open(file_path, 'rb') as f:
        put_resp = http_requests.put(
            upload_data['url'],
            data=f,
            headers={'Content-Type': 'application/pdf'},
        )
    if not put_resp.ok:
        raise Exception(f'File upload failed: {put_resp.status_code}')

    # 3. Create object record
    obj_resp = http_requests.post(
        f'{VERTESIA_API_BASE}/objects',
        headers=headers,
        json={
            'name': filename,
            'content': {
                'source': upload_data['id'],
                'type': 'application/pdf',
                'name': filename,
            },
        },
    )
    if not obj_resp.ok:
        raise Exception(f'Object create failed: {obj_resp.status_code} {obj_resp.text[:200]}')
    doc = obj_resp.json()
    doc_id = doc['id']

    # 4. Add to collection (optional)
    if collection_id:
        http_requests.post(
            f'{VERTESIA_API_BASE}/collections/{collection_id}/members',
            headers=headers,
            json={'id': doc_id},
        )

    return doc_id


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------
def job_split_only(job, pdf_path, output_dir, client_name='', project=''):
    """Background worker for /split — split PDF, create ZIP for download."""
    try:
        parts = [p for p in [client_name, project] if p]
        naming_prefix = '_'.join(parts).replace(' ', '_') if parts else None

        result = process_pdf(pdf_path, job.batch_size, job)
        bf = split_pdf_into_batches(
            pdf_path, result['manifest']['batches'], output_dir, naming_prefix,
        )

        manifest = {**result['manifest']}
        manifest['batches'] = [{k: v for k, v in b.items() if k != 'path'} for b in bf]

        mp = os.path.join(output_dir, 'manifest.json')
        with open(mp, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        zp = os.path.join(output_dir, 'batches.zip')
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(mp, 'manifest.json')
            for b in bf:
                z.write(b['path'], b['filename'])

        with jobs_lock:
            job.status = "done"
            job.progress = 100
            job.phase = "done"
            job.result_path = zp
            job.output_dir = output_dir
            job.manifest = manifest

    except Exception as e:
        import traceback; traceback.print_exc()
        with jobs_lock:
            job.status = "error"
            job.error = str(e)
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def job_split_and_upload(job, pdf_path, output_dir, vertesia_jwt,
                         collection_id=None, client_name='', project=''):
    """Background worker for /split-and-upload — split PDF, upload to Vertesia."""
    try:
        # Build naming prefix: "ClientName_Project" or just "ClientName" or fallback
        parts = [p for p in [client_name, project] if p]
        naming_prefix = '_'.join(parts).replace(' ', '_') if parts else None

        result = process_pdf(pdf_path, job.batch_size, job)
        bf = split_pdf_into_batches(
            pdf_path, result['manifest']['batches'], output_dir, naming_prefix,
        )

        # Upload phase
        job.phase = "uploading"
        job.progress = 92

        document_ids = []
        for i, batch in enumerate(bf):
            doc_id = upload_to_vertesia(
                batch['path'], batch['filename'], vertesia_jwt, collection_id,
            )
            document_ids.append({
                'batchNumber': batch['batchNumber'],
                'documentId': doc_id,
                'filename': batch['filename'],
                'customerCount': batch['customerCount'],
                'pageCount': batch['pageCount'],
                'pageStart': batch['pageStart'],
                'pageEnd': batch['pageEnd'],
                'customers': batch['customers'],
            })
            job.progress = 92 + int((i + 1) / len(bf) * 8)

        # Build final manifest with doc IDs
        manifest = {**result['manifest']}
        manifest['batches'] = [{k: v for k, v in b.items() if k != 'path'} for b in bf]
        manifest['documentIds'] = document_ids
        if client_name:
            manifest['clientName'] = client_name
        if project:
            manifest['project'] = project

        with jobs_lock:
            job.status = "done"
            job.progress = 100
            job.phase = "done"
            job.output_dir = output_dir
            job.manifest = manifest

    except Exception as e:
        import traceback; traceback.print_exc()
        with jobs_lock:
            job.status = "error"
            job.error = str(e)
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return jsonify({
        'service': 'PDF Chunker',
        'endpoints': {
            'POST /split': 'Split PDF → poll + download batches',
            'POST /split-and-upload': 'Split PDF + upload to Vertesia → doc IDs',
            'GET /status/<jobId>': 'Poll progress (returns manifest when done)',
            'GET /batch/<jobId>/<batchNum>': 'Download individual batch PDF',
            'GET /download/<jobId>': 'Download all batches as ZIP',
            'POST /diagnose': 'Upload PDF → get boundary detection diagnostics (no split)',
        },
    })


@app.route('/split', methods=['POST', 'OPTIONS'])
def split():
    if request.method == 'OPTIONS':
        return '', 204
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    batch_size = int(request.form.get('batch_size', 12))
    client_name = request.form.get('client_name', '').strip()
    project = request.form.get('project', '').strip()
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, f.filename, batch_size)

    td = tempfile.mkdtemp()
    od = os.path.join(td, 'batches')
    os.makedirs(od)
    pp = os.path.join(td, 'input.pdf')
    f.save(pp)

    with jobs_lock:
        jobs[job_id] = job

    threading.Thread(
        target=job_split_only,
        args=(job, pp, od, client_name, project),
        daemon=True,
    ).start()
    return jsonify({'jobId': job_id, 'status': 'pending'})


@app.route('/split-and-upload', methods=['POST', 'OPTIONS'])
def split_and_upload():
    """Split PDF and upload batches directly to Vertesia. Returns document IDs."""
    if request.method == 'OPTIONS':
        return '', 204
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    vertesia_jwt = request.form.get('vertesia_jwt')
    if not vertesia_jwt:
        return jsonify({'error': 'vertesia_jwt is required'}), 400

    batch_size = int(request.form.get('batch_size', 12))
    collection_id = request.form.get('collection_id')
    client_name = request.form.get('client_name', '').strip()
    project = request.form.get('project', '').strip()

    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, f.filename, batch_size)

    td = tempfile.mkdtemp()
    od = os.path.join(td, 'batches')
    os.makedirs(od)
    pp = os.path.join(td, 'input.pdf')
    f.save(pp)

    with jobs_lock:
        jobs[job_id] = job

    threading.Thread(
        target=job_split_and_upload,
        args=(job, pp, od, vertesia_jwt, collection_id, client_name, project),
        daemon=True,
    ).start()

    return jsonify({'jobId': job_id, 'status': 'pending'})


@app.route('/status/<jid>')
def status(jid):
    with jobs_lock:
        j = jobs.get(jid)
    return jsonify(j.to_dict()) if j else (jsonify({'error': 'Not found'}), 404)


@app.route('/batch/<jid>/<int:batch_num>')
def get_batch(jid, batch_num):
    with jobs_lock:
        j = jobs.get(jid)
    if not j:
        return jsonify({'error': 'Job not found'}), 404
    if j.status != 'done':
        return jsonify({'error': 'Job not complete'}), 400
    if not j.output_dir:
        return jsonify({'error': 'Output not available'}), 404
    batch_file = os.path.join(j.output_dir, f"batch_{batch_num:03d}.pdf")
    if not os.path.exists(batch_file):
        return jsonify({'error': f'Batch {batch_num} not found'}), 404
    return send_file(batch_file, mimetype='application/pdf', as_attachment=True,
                     download_name=f"batch_{batch_num:03d}.pdf")


@app.route('/download/<jid>')
def download(jid):
    with jobs_lock:
        j = jobs.get(jid)
    if not j:
        return jsonify({'error': 'Not found'}), 404
    if j.status != 'done':
        return jsonify({'error': 'Not complete'}), 400
    if not j.result_path or not os.path.exists(j.result_path):
        return jsonify({'error': 'File missing'}), 404
    return send_file(j.result_path, mimetype='application/zip', as_attachment=True,
                     download_name=f"batches_{j.filename.replace('.pdf', '')}.zip")


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'ocr_available': HAS_OCR})


@app.route('/diagnose', methods=['GET', 'POST', 'OPTIONS'])
def diagnose():
    """Upload a PDF and get boundary detection diagnostics without splitting."""
    if request.method == 'OPTIONS':
        return '', 204

    if request.method == 'GET':
        return '''<!DOCTYPE html>
<html><head><title>PDF Chunker - Diagnose</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h1 { color: #333; }
  .upload-box { background: white; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 20px 0; }
  input[type=file] { margin: 12px 0; }
  button { background: #2563eb; color: white; border: none; padding: 10px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #93c5fd; cursor: wait; }
  #status { margin: 16px 0; color: #666; }
  pre { background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; max-height: 600px; overflow-y: auto; white-space: pre-wrap; }
</style></head>
<body>
  <h1>PDF Chunker - Diagnose</h1>
  <p>Upload a PDF to see what the boundary detector finds. No splitting or uploading happens.</p>
  <div class="upload-box">
    <input type="file" id="file" accept=".pdf"><br>
    <button onclick="run()" id="btn">Run Diagnostics</button>
    <div id="status"></div>
  </div>
  <pre id="result" style="display:none"></pre>
  <script>
    async function run() {
      const file = document.getElementById('file').files[0];
      if (!file) return alert('Pick a PDF first');
      const btn = document.getElementById('btn');
      const status = document.getElementById('status');
      const result = document.getElementById('result');
      btn.disabled = true;
      status.textContent = 'Running detection (OCR may take a minute)...';
      result.style.display = 'none';
      const fd = new FormData();
      fd.append('file', file);
      try {
        const resp = await fetch('/diagnose', { method: 'POST', body: fd });
        const data = await resp.json();
        result.textContent = JSON.stringify(data, null, 2);
        result.style.display = 'block';
        status.textContent = resp.ok
          ? 'Done - ' + data.customersFound + ' customers found (' + data.mode + ' mode)'
          : 'Error - ' + (data.error || 'unknown');
      } catch (e) {
        status.textContent = 'Request failed: ' + e.message;
      }
      btn.disabled = false;
    }
  </script>
</body></html>''', 200, {'Content-Type': 'text/html'}

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    td = tempfile.mkdtemp()
    pp = os.path.join(td, 'input.pdf')
    f.save(pp)

    job = Job('diag', f.filename, 1)
    try:
        custs = detect_boundaries_text(pp, job)
        if custs:
            mode = 'text'
        else:
            mode = 'ocr'
            if HAS_OCR:
                custs = detect_boundaries_ocr(pp, job)
            else:
                custs = []

        result = {
            'filename': f.filename,
            'mode': mode,
            'customersFound': len(custs) if custs else 0,
            'customers': [
                {
                    'index': c['index'], 'name': c['name'],
                    'cityStateZip': c['cityStateZip'],
                    'pageStart': c['pageStart'], 'pageEnd': c['pageEnd'],
                    'pageCount': c['pageCount'], 'confidence': c['confidence'],
                }
                for c in (custs or [])
            ],
            'diagnostics': job.diagnostics,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'diagnostics': job.diagnostics,
        }), 500
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"PDF Chunker on :{port}")
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', '') == 'true')
