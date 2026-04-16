"""
PDF Chunker - Customer Boundary Detection + Supabase Upload

Endpoints:
  POST /split                    - Split only, returns jobId (poll + download batches)
  POST /split-and-upload         - Split + upload to Supabase After Samples collection
  GET  /status/<jobId>           - Poll job progress (includes diagnostics on error)
  GET  /batch/<jobId>/<batchNum> - Download individual batch PDF
  GET  /download/<jobId>         - Download all batches as ZIP
  GET|POST /diagnose             - Boundary detection diagnostics
  GET  /health                   - Health check

Strategy: text extraction first (~2s for 600 pages), OCR fallback if needed.
Boundary detection: address-frequency classification.
"""

import os, re, json, tempfile, uuid, threading, time, zipfile, io, shutil
from collections import Counter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import fitz  # PyMuPDF
from supabase import create_client as create_supabase_client

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# ---------------------------------------------------------------------------
# Supabase config
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
STORAGE_BUCKET = 'documents'

sb = create_supabase_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Supabase upload (replaces Vertesia)
# ---------------------------------------------------------------------------
def upload_to_supabase(file_path, filename, collection_id=None):
    """Upload a PDF to Supabase storage, create object record, add to collection."""
    object_id = str(uuid.uuid4())
    storage_path = f"{object_id}/{filename}"

    with open(file_path, 'rb') as f:
        sb.storage.from_(STORAGE_BUCKET).upload(
            storage_path, f.read(),
            file_options={"content-type": "application/pdf"}
        )

    sb.table('objects').insert({
        'id': object_id,
        'name': filename,
        'mime_type': 'application/pdf',
        'storage_path': storage_path
    }).execute()

    if collection_id:
        sb.table('collection_members').insert({
            'collection_id': collection_id,
            'object_id': object_id
        }).execute()

    return object_id


def find_after_samples_collection(client_name):
    """Find the After Samples collection for a client by group_name.
    Handles: 'Aftersamples', 'After Samples', 'Raw Aftersamples', etc."""
    for pattern in ['%aftersample%', '%after sample%', '%after%sample%']:
        result = sb.table('collections').select('id').eq('group_name', client_name).ilike('type_name', pattern).execute()
        if result.data and len(result.data) > 0:
            print(f"[UPLOAD] Found collection via pattern '{pattern}': {result.data[0]['id']}")
            return result.data[0]['id']
    return None


# ---------------------------------------------------------------------------
# Job storage
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
        self.output_dir = None
        self.manifest = None
        self.created_at = time.time()
        self.mode_used = None
        self.diagnostics = None

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
# Shared helpers (unchanged from original)
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
MIN_TEXT_CHARS = 200


def is_likely_person_name(t):
    if not t or len(t) < 3 or len(t) > 80: return False
    if any(p.search(t) for p in NON_NAME_PATTERNS): return False
    if len(t.split()) < 2: return False
    return sum(1 for c in t if c.isalpha() or c in " '-") / len(t) >= 0.8

def clean_line(t):
    if not t: return t
    c = NOISE_PREFIXES.sub('', t).strip()
    c = re.sub(r'^[^A-Za-z0-9]+', '', c).strip()
    return c or t

def _extract_address_from_lines(lines, i):
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
    lo = ln = None
    for line in text.split('\n'):
        if 'old' in line.lower() and 'loan' in line.lower():
            m = LOAN_RE.search(line)
            if m: lo = m.group(1)
        if 'new' in line.lower() and 'loan' in line.lower():
            m = LOAN_RE.search(line)
            if m: ln = m.group(1)
    return lo, ln

def _classify_corporate_addresses(csz_pages, total_pages):
    corp = set()
    for csz, pages in csz_pages.items():
        pages_sorted = sorted(set(pages))
        runs = 1
        for i in range(1, len(pages_sorted)):
            if pages_sorted[i] - pages_sorted[i - 1] > 1: runs += 1
        if runs > 1 and len(pages_sorted) / total_pages > 0.2:
            corp.add(csz)
    return corp

def _safe_customer_filename(name):
    if not name: return None
    safe = re.sub(r'[^\w\s\-]', '', name).strip()
    safe = re.sub(r'\s+', '_', safe)
    return safe if safe and safe != 'Not_detected' else None


# ---------------------------------------------------------------------------
# Text extraction (fast path) — UNCHANGED
# ---------------------------------------------------------------------------
def detect_boundaries_text(pdf_path, job=None):
    doc = fitz.open(pdf_path)
    total = doc.page_count
    print(f"[DETECT-TEXT] Starting text scan: {total} pages")
    if job:
        job.total_pages = total; job.phase = "text_scan"; job.status = "processing"

    page_text_lengths = [len(doc[p].get_text().strip()) for p in range(total)]
    pages_with_substance = sum(1 for l in page_text_lengths if l >= MIN_TEXT_CHARS)
    substance_ratio = pages_with_substance / total if total else 0
    print(f"[DETECT-TEXT] Text quality: {pages_with_substance}/{total} pages >= {MIN_TEXT_CHARS} chars ({substance_ratio:.0%})")

    if substance_ratio < 0.5:
        print(f"[DETECT-TEXT] Text layer too thin. Falling back to OCR")
        doc.close()
        if job: job.diagnostics = {'stage': 'text_quality_gate', 'reason': 'text_layer_too_thin'}
        return None

    page_addr_details = {}
    csz_pages = {}
    for p in range(total):
        text = doc[p].get_text()
        if not text.strip(): page_addr_details[p] = []; continue
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        addrs = []; seen = set()
        for i, line in enumerate(lines):
            m = CITY_STATE_ZIP_RE.match(line)
            if not m or m.group(2) not in US_STATES: continue
            city, state, zc = m.group(1).strip(), m.group(2), m.group(3)
            key = f"{city.upper()}, {state} {zc}"
            if key in seen: continue
            seen.add(key)
            name, street = _extract_address_from_lines(lines, i)
            addrs.append({'csz_key': key, 'city': city, 'state': state, 'zip': zc, 'name': name, 'street': street or ''})
            csz_pages.setdefault(key, []).append(p)
        page_addr_details[p] = addrs
        if job and p % 50 == 0: job.progress = int((p / total) * 30)

    if not csz_pages:
        print(f"[DETECT-TEXT] No addresses found. Falling back to OCR"); doc.close(); return None

    corp = _classify_corporate_addresses(csz_pages, total)
    print(f"[DETECT-TEXT] Addresses: {len(csz_pages)} unique, Corporate: {len(corp)}, Customer: {len(csz_pages) - len(corp)}")
    if job: job.phase = "text_extract"; job.progress = 35

    page_customer = [None] * total; page_detail = [None] * total
    for p in range(total):
        for a in page_addr_details[p]:
            if a['csz_key'] not in corp and a['name']:
                page_customer[p] = a['csz_key']; page_detail[p] = a; break

    boundaries = []; current = None
    for p in range(total):
        csz = page_customer[p]
        if csz and csz != current: boundaries.append(p); current = csz

    if not boundaries:
        print(f"[DETECT-TEXT] No boundaries found. Falling back to OCR"); doc.close(); return None
    if len(boundaries) == 1 and total > 10:
        print(f"[DETECT-TEXT] Only 1 boundary in {total}-page doc. Falling back to OCR"); doc.close(); return None

    print(f"[DETECT-TEXT] Boundaries: {len(boundaries)} at pages {[b+1 for b in boundaries]}")
    if job:
        job.diagnostics = {'stage': 'text_boundary_detect', 'total_pages': total, 'boundaries_found': len(boundaries), 'boundary_pages': [b+1 for b in boundaries]}

    customers = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else total - 1
        d = page_detail[start]
        lo = ln = None
        for scan_p in range(start, min(start + 3, total)):
            plo, pln = _extract_loans_from_text(doc[scan_p].get_text())
            if plo and not lo: lo = plo
            if pln and not ln: ln = pln
        customers.append({
            'name': d['name'] or '(Not detected)', 'street': d['street'],
            'city': d['city'], 'state': d['state'], 'zip': d['zip'],
            'cityStateZip': d['csz_key'], 'loanOld': lo, 'loanNew': ln,
            'pageStart': start + 1, 'pageEnd': end + 1, 'pageCount': end - start + 1,
            'confidence': 'strong' if d['name'] and d['street'] else 'medium' if d['name'] else 'weak',
            'addressLines': [l for l in [d['name'], d['street'], d['csz_key']] if l],
            'index': idx + 1,
        })
        if job: job.customers_found = len(customers); job.progress = 35 + int((idx / len(boundaries)) * 55)

    doc.close()
    return customers


# ---------------------------------------------------------------------------
# OCR fallback — UNCHANGED (abbreviated for space, same logic as original)
# ---------------------------------------------------------------------------
def detect_boundaries_ocr(pdf_path, job=None):
    if not HAS_OCR:
        raise RuntimeError("No text layer found and OCR (pytesseract) not installed")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    doc = fitz.open(pdf_path); total = doc.page_count
    print(f"[DETECT-OCR] Starting OCR scan: {total} pages")
    if job: job.total_pages = total; job.status = "processing"; job.phase = "rendering"

    imgs = {}
    for p in range(total):
        pg = doc[p]; r = pg.rect; clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + (r.height * 0.6))
        imgs[p] = pg.get_pixmap(dpi=150, clip=clip).tobytes("png")
        if job and p % 20 == 0: job.pages_complete = p + 1; job.progress = int((p / total) * 15)
    doc.close()
    if job: job.phase = "ocr"

    page_ocr_addrs = {}; csz_pages = {}; ocr_lock = threading.Lock(); done = [0]
    def ocr_page(p):
        img = Image.open(io.BytesIO(imgs[p]))
        text = pytesseract.image_to_string(img, config='--psm 6')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        addrs = []; seen = set()
        for i, line in enumerate(lines):
            lc = clean_line(line); m = CITY_STATE_ZIP_RE.match(lc)
            if not m or m.group(2).upper() not in US_STATES: continue
            city, state, zc = m.group(1).strip(), m.group(2).upper(), m.group(3)
            key = f"{city.upper()}, {state} {zc}"
            if key in seen: continue
            seen.add(key)
            name, street = _extract_address_from_lines([clean_line(z) for z in lines], i)
            addrs.append({'csz_key': key, 'city': city, 'state': state, 'zip': zc, 'name': name, 'street': street or ''})
        with ocr_lock:
            page_ocr_addrs[p] = addrs
            for a in addrs: csz_pages.setdefault(a['csz_key'], []).append(p)
        done[0] += 1
        if job and done[0] % 10 == 0: job.pages_complete = done[0]; job.progress = 15 + int((done[0] / total) * 55)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for fut in as_completed([ex.submit(ocr_page, p) for p in range(total)]): fut.result()

    corp = _classify_corporate_addresses(csz_pages, total)
    print(f"[DETECT-OCR] Complete. Corporate: {len(corp)}, Customer: {len(csz_pages) - len(corp)}")
    if job: job.phase = "text_extract"; job.progress = 75

    page_customer = [None] * total; page_detail = [None] * total
    for p in range(total):
        for a in page_ocr_addrs.get(p, []):
            if a['csz_key'] not in corp: page_customer[p] = a['csz_key']; page_detail[p] = a; break

    boundaries = []; current = None
    for p in range(total):
        csz = page_customer[p]
        if csz and csz != current: boundaries.append(p); current = csz

    print(f"[DETECT-OCR] Boundaries: {len(boundaries)}")
    customers = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else total - 1
        d = page_detail[start]
        customers.append({
            'name': d['name'] or '(Not detected)', 'street': d['street'],
            'city': d['city'], 'state': d['state'], 'zip': d['zip'],
            'cityStateZip': d['csz_key'], 'loanOld': None, 'loanNew': None,
            'pageStart': start + 1, 'pageEnd': end + 1, 'pageCount': end - start + 1,
            'confidence': 'strong' if d['name'] and d['street'] else 'medium' if d['name'] else 'weak',
            'addressLines': [l for l in [d['name'], d['street'], d['csz_key']] if l],
            'index': idx + 1,
        })
        if job: job.customers_found = len(customers)
    return customers


# ---------------------------------------------------------------------------
# Main processor — UNCHANGED
# ---------------------------------------------------------------------------
def process_pdf(pdf_path, batch_size=1, job=None):
    start = time.time()
    if job: job.phase = "text_scan"; job.status = "processing"
    custs = detect_boundaries_text(pdf_path, job)
    if custs:
        mode = 'text'
        if job: job.mode_used = 'text'
    else:
        mode = 'ocr'
        if job: job.phase = "ocr_fallback"; job.mode_used = 'ocr'
        custs = detect_boundaries_ocr(pdf_path, job)
    if not custs:
        raise RuntimeError(f"No customer boundaries detected (mode={mode})")
    elapsed = time.time() - start
    print(f"[PROCESS] {len(custs)} customers via {mode} in {elapsed:.1f}s")
    if job: job.phase = "batching"; job.progress = 92; job.customers_found = len(custs)

    doc = fitz.open(pdf_path); total = doc.page_count; doc.close()
    batches = []
    for i in range(0, len(custs), batch_size):
        bc = custs[i:i + batch_size]
        batches.append({
            'batchNumber': len(batches) + 1, 'customerCount': len(bc),
            'pageStart': bc[0]['pageStart'], 'pageEnd': bc[-1]['pageEnd'],
            'pageCount': bc[-1]['pageEnd'] - bc[0]['pageStart'] + 1,
            'customers': bc,
        })
    manifest = {
        'totalPages': total, 'totalCustomers': len(custs), 'totalBatches': len(batches),
        'batchSize': batch_size, 'mode': mode, 'elapsedSeconds': round(elapsed, 2),
        'batches': batches,
    }
    return {'manifest': manifest, 'customers': custs}


# ---------------------------------------------------------------------------
# PDF splitting
# ---------------------------------------------------------------------------
def split_pdf_into_batches(pdf_path, batches, output_dir, client_name='', project=''):
    doc = fitz.open(pdf_path)
    files = []
    for b in batches:
        bd = fitz.open()
        bd.insert_pdf(doc, from_page=b['pageStart'] - 1, to_page=b['pageEnd'] - 1)
        is_single = (b['customerCount'] == 1)
        if is_single:
            cust = b['customers'][0]
            cust_name = _safe_customer_filename(cust.get('name', ''))
            loan = cust.get('loanNew') or cust.get('loanOld') or ''
            # Format: "BRUNGARDT_JOSHUA_3200066433.pdf"
            parts = []
            if cust_name: parts.append(cust_name)
            if loan: parts.append(loan)
            fn = '_'.join(parts) + '.pdf' if parts else f"customer_{b['batchNumber']:03d}.pdf"
        else:
            fn = f"batch_{b['batchNumber']:03d}.pdf"
        fn = re.sub(r'[<>:"/\\|?*]', '', fn)
        fp = os.path.join(output_dir, fn)
        bd.save(fp); bd.close()
        files.append({**b, 'filename': fn, 'path': fp})
    doc.close()
    return files


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------
def job_split_only(job, pdf_path, output_dir, client_name='', project=''):
    try:
        result = process_pdf(pdf_path, job.batch_size, job)
        bf = split_pdf_into_batches(pdf_path, result['manifest']['batches'], output_dir, client_name, project)
        manifest = {**result['manifest']}
        manifest['batches'] = [{k: v for k, v in b.items() if k != 'path'} for b in bf]
        if client_name: manifest['clientName'] = client_name
        if project: manifest['project'] = project

        mp = os.path.join(output_dir, 'manifest.json')
        with open(mp, 'w') as mf: json.dump(manifest, mf, indent=2, default=str)
        zp = os.path.join(output_dir, 'batches.zip')
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(mp, 'manifest.json')
            for b in bf: z.write(b['path'], b['filename'])

        with jobs_lock:
            job.status = "done"; job.progress = 100; job.phase = "done"
            job.result_path = zp; job.output_dir = output_dir; job.manifest = manifest
    except Exception as e:
        import traceback; traceback.print_exc()
        with jobs_lock: job.status = "error"; job.error = str(e)
    finally:
        if os.path.exists(pdf_path): os.unlink(pdf_path)


def job_split_and_upload_supabase(job, pdf_path, output_dir, client_name='', project=''):
    """Split PDF and upload each customer's document to Supabase."""
    try:
        result = process_pdf(pdf_path, job.batch_size, job)
        bf = split_pdf_into_batches(pdf_path, result['manifest']['batches'], output_dir, client_name, project)

        job.phase = "uploading"; job.progress = 92

        # Find the After Samples collection for this client
        collection_id = find_after_samples_collection(client_name) if client_name else None
        if collection_id:
            print(f"[UPLOAD] Found After Samples collection: {collection_id}")
        else:
            print(f"[UPLOAD] No After Samples collection found for '{client_name}' — uploading without collection")

        document_ids = []
        for i, batch in enumerate(bf):
            print(f"[UPLOAD] Uploading {batch['filename']}...")
            doc_id = upload_to_supabase(batch['path'], batch['filename'], collection_id)
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
            print(f"[UPLOAD] {batch['filename']} → {doc_id}")

        manifest = {**result['manifest']}
        manifest['batches'] = [{k: v for k, v in b.items() if k != 'path'} for b in bf]
        manifest['documentIds'] = document_ids
        if client_name: manifest['clientName'] = client_name
        if project: manifest['project'] = project

        with jobs_lock:
            job.status = "done"; job.progress = 100; job.phase = "done"
            job.output_dir = output_dir; job.manifest = manifest

        print(f"[UPLOAD] Complete: {len(document_ids)} documents uploaded to Supabase")

    except Exception as e:
        import traceback; traceback.print_exc()
        with jobs_lock: job.status = "error"; job.error = str(e)
    finally:
        if os.path.exists(pdf_path): os.unlink(pdf_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return jsonify({
        'service': 'PDF Chunker (Supabase)',
        'endpoints': {
            'POST /split': 'Split PDF → poll + download batches',
            'POST /split-and-upload': 'Split PDF + upload to Supabase',
            'GET /status/<jobId>': 'Poll progress',
            'GET /batch/<jobId>/<batchNum>': 'Download individual batch PDF',
            'GET /download/<jobId>': 'Download all batches as ZIP',
            'GET|POST /diagnose': 'Boundary detection diagnostics',
        },
    })


@app.route('/split', methods=['POST', 'OPTIONS'])
def split():
    if request.method == 'OPTIONS': return '', 204
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if f.filename == '': return jsonify({'error': 'No file selected'}), 400

    batch_size = int(request.form.get('batch_size', 1))
    client_name = request.form.get('client_name', '').strip()
    project = request.form.get('project', '').strip()

    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, f.filename, batch_size)
    td = tempfile.mkdtemp(); od = os.path.join(td, 'batches'); os.makedirs(od)
    pp = os.path.join(td, 'input.pdf'); f.save(pp)

    with jobs_lock: jobs[job_id] = job
    threading.Thread(target=job_split_only, args=(job, pp, od, client_name, project), daemon=True).start()
    return jsonify({'jobId': job_id, 'status': 'pending'})


@app.route('/split-and-upload', methods=['POST', 'OPTIONS'])
def split_and_upload():
    """Split PDF and upload individual customer PDFs to Supabase."""
    if request.method == 'OPTIONS': return '', 204
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if f.filename == '': return jsonify({'error': 'No file selected'}), 400

    # batch_size=1 by default — each customer gets their own PDF
    batch_size = int(request.form.get('batch_size', 1))
    client_name = request.form.get('client_name', '').strip()
    project = request.form.get('project', '').strip()

    if not client_name:
        return jsonify({'error': 'client_name is required to find the After Samples collection'}), 400

    print(f"[SPLIT-UPLOAD] client_name={client_name}, project={project}, batch_size={batch_size}, file={f.filename}")

    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, f.filename, batch_size)
    td = tempfile.mkdtemp(); od = os.path.join(td, 'batches'); os.makedirs(od)
    pp = os.path.join(td, 'input.pdf'); f.save(pp)

    with jobs_lock: jobs[job_id] = job
    threading.Thread(
        target=job_split_and_upload_supabase,
        args=(job, pp, od, client_name, project),
        daemon=True
    ).start()

    return jsonify({'jobId': job_id, 'status': 'pending'})


@app.route('/status/<jid>')
def status(jid):
    with jobs_lock: j = jobs.get(jid)
    return jsonify(j.to_dict()) if j else (jsonify({'error': 'Not found'}), 404)

@app.route('/batch/<jid>/<int:batch_num>')
def get_batch(jid, batch_num):
    with jobs_lock: j = jobs.get(jid)
    if not j: return jsonify({'error': 'Not found'}), 404
    if j.status != 'done': return jsonify({'error': 'Not complete'}), 400
    if not j.output_dir: return jsonify({'error': 'No output'}), 404
    bf = os.path.join(j.output_dir, f"batch_{batch_num:03d}.pdf")
    if not os.path.exists(bf): return jsonify({'error': f'Batch {batch_num} not found'}), 404
    return send_file(bf, mimetype='application/pdf', as_attachment=True)

@app.route('/download/<jid>')
def download(jid):
    with jobs_lock: j = jobs.get(jid)
    if not j: return jsonify({'error': 'Not found'}), 404
    if j.status != 'done': return jsonify({'error': 'Not complete'}), 400
    if not j.result_path or not os.path.exists(j.result_path): return jsonify({'error': 'Missing'}), 404
    return send_file(j.result_path, mimetype='application/zip', as_attachment=True)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'ocr_available': HAS_OCR, 'storage': 'supabase'})


@app.route('/diagnose', methods=['GET', 'POST', 'OPTIONS'])
def diagnose():
    if request.method == 'OPTIONS': return '', 204
    if request.method == 'GET':
        return '''<!DOCTYPE html><html><head><title>PDF Chunker - Diagnose</title>
<style>body{font-family:system-ui;max-width:800px;margin:40px auto;padding:0 20px;background:#f5f5f5}h1{color:#333}.upload-box{background:white;padding:24px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);margin:20px 0}button{background:#2563eb;color:white;border:none;padding:10px 24px;border-radius:6px;cursor:pointer}button:disabled{background:#93c5fd;cursor:wait}pre{background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;overflow:auto;font-size:13px;max-height:600px;white-space:pre-wrap}</style></head>
<body><h1>PDF Chunker - Diagnose</h1><p>Upload a PDF to see boundary detection results.</p>
<div class="upload-box"><input type="file" id="file" accept=".pdf"><br><br><button onclick="run()" id="btn">Run</button><div id="status" style="margin:12px 0;color:#666"></div></div><pre id="result" style="display:none"></pre>
<script>async function run(){const f=document.getElementById('file').files[0];if(!f)return alert('Pick a PDF');const btn=document.getElementById('btn'),st=document.getElementById('status'),res=document.getElementById('result');btn.disabled=true;st.textContent='Running...';res.style.display='none';const fd=new FormData();fd.append('file',f);try{const r=await fetch('/diagnose',{method:'POST',body:fd});const d=await r.json();res.textContent=JSON.stringify(d,null,2);res.style.display='block';st.textContent=r.ok?d.customersFound+' customers ('+d.mode+')':'Error';}catch(e){st.textContent='Failed: '+e.message;}btn.disabled=false;}</script></body></html>''', 200, {'Content-Type': 'text/html'}

    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    td = tempfile.mkdtemp(); pp = os.path.join(td, 'input.pdf'); f.save(pp)
    job = Job('diag', f.filename, 1)
    try:
        custs = detect_boundaries_text(pp, job)
        if custs: mode = 'text'
        else:
            mode = 'ocr'
            custs = detect_boundaries_ocr(pp, job) if HAS_OCR else []
        return jsonify({'filename': f.filename, 'mode': mode, 'customersFound': len(custs) if custs else 0,
            'customers': [{'index': c['index'], 'name': c['name'], 'cityStateZip': c['cityStateZip'],
                'pageStart': c['pageStart'], 'pageEnd': c['pageEnd'], 'pageCount': c['pageCount'],
                'confidence': c['confidence'], 'loanNew': c.get('loanNew'), 'loanOld': c.get('loanOld')} for c in (custs or [])],
            'diagnostics': job.diagnostics})
    except Exception as e:
        return jsonify({'error': str(e), 'diagnostics': job.diagnostics}), 500
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"PDF Chunker (Supabase) on :{port}")
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', '') == 'true')
