"""
PDF Chunker - Customer Boundary Detection + Vertesia Upload
Runs as a microservice. QA Platform sends PDFs here.

Endpoints:
  POST /split                    — Split only, returns jobId (poll + download batches)
  POST /split-and-upload         — Split + upload batches to Vertesia, returns doc IDs
  GET  /status/<jobId>           — Poll job progress
  GET  /batch/<jobId>/<batchNum> — Download individual batch PDF
  GET  /download/<jobId>         — Download all batches as ZIP
  GET  /health                   — Health check

Strategy: text extraction first (~2s for 600 pages), OCR fallback if needed.
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
# Helpers
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
    r'^[\d\s\-\+]+$', r'Corporate Drive', r'Lake Zurich',
]]
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


# ---------------------------------------------------------------------------
# Text extraction (fast path — ~1-2 sec for 600 pages)
# ---------------------------------------------------------------------------
def detect_boundaries_text(pdf_path, job=None):
    doc = fitz.open(pdf_path)
    total = doc.page_count
    if job:
        job.total_pages = total
        job.phase = "text_scan"
        job.status = "processing"

    bpages = []
    for i in range(total):
        if "NOTICE OF SERVICING TRANSFER" in doc[i].get_text():
            bpages.append(i)
        if job and i % 50 == 0:
            job.progress = int((i / total) * 30)

    if len(bpages) < 2:
        doc.close()
        return None

    if job:
        job.phase = "text_extract"
        job.progress = 35

    customers = []
    for idx, pg in enumerate(bpages):
        text = doc[pg].get_text()
        lo = ln = None
        for line in text.split('\n'):
            if 'Old Loan Number' in line:
                lo = line.split(':')[-1].strip()
            if 'New Loan Number' in line:
                ln = line.split(':')[-1].strip()

        name = street = csz = city = state = zc = None
        addr = []
        for off in range(1, 4):
            if pg + off >= total:
                break
            lines = [l.strip() for l in doc[pg + off].get_text().split('\n') if l.strip()]
            for j, line in enumerate(lines):
                if 'next payment' in line.lower():
                    parsed = _parse_address_block(lines[j + 1:])
                    if parsed:
                        name = parsed['name']
                        street = parsed.get('street')
                        city = parsed.get('city')
                        state = parsed.get('state')
                        zc = parsed.get('zip')
                        csz = parsed.get('cityStateZip')
                        addr = parsed.get('addressLines', [])
                    break
            if name:
                break

        pe = bpages[idx + 1] if idx + 1 < len(bpages) else total
        customers.append({
            'name': name or '(Not detected)',
            'street': street or '',
            'city': city or '',
            'state': state or 'XX',
            'zip': zc or '00000',
            'cityStateZip': csz or '',
            'loanOld': lo,
            'loanNew': ln,
            'pageStart': pg + 1,
            'pageEnd': pe,
            'pageCount': pe - pg,
            'confidence': 'strong' if name and street else 'medium' if name else 'weak',
            'addressLines': addr,
            'index': idx + 1,
        })
        if job:
            job.customers_found = len(customers)
            job.progress = 35 + int((idx / len(bpages)) * 55)

    doc.close()
    return customers


def _parse_address_block(lines):
    if not lines:
        return None
    for i, line in enumerate(lines):
        m = CITY_STATE_ZIP_RE.match(line)
        if m and m.group(2) in US_STATES:
            city, state, zc = m.group(1).strip(), m.group(2), m.group(3)
            if 'ZURICH' in city.upper() or 'GREENVILLE' in city.upper():
                continue
            r = {
                'city': city, 'state': state, 'zip': zc,
                'cityStateZip': f"{city}, {state} {zc}", 'addressLines': [],
            }
            if i >= 1 and STREET_RE.match(lines[i - 1]):
                r['street'] = lines[i - 1]
                if i >= 2 and is_likely_person_name(lines[i - 2]):
                    r['name'] = (
                        f"{lines[i - 3]} / {lines[i - 2]}"
                        if i >= 3 and is_likely_person_name(lines[i - 3])
                        else lines[i - 2]
                    )
                elif i >= 2:
                    r['name'] = lines[i - 2]
            elif i >= 1 and is_likely_person_name(lines[i - 1]):
                r['name'] = lines[i - 1]
            if r.get('name'):
                r['addressLines'] = [
                    l for l in [r.get('name'), r.get('street'), r['cityStateZip']] if l
                ]
                return r
    return None


# ---------------------------------------------------------------------------
# OCR fallback (if text extraction fails)
# ---------------------------------------------------------------------------
def detect_boundaries_ocr(pdf_path, job=None):
    if not HAS_OCR:
        raise RuntimeError("No text layer found and OCR (pytesseract) not installed")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    doc = fitz.open(pdf_path)
    total = doc.page_count
    if job:
        job.total_pages = total
        job.status = "processing"
        job.phase = "rendering"

    imgs = {}
    for p in range(total):
        pg = doc[p]
        r = pg.rect
        clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + (r.height * 0.4))
        imgs[p] = pg.get_pixmap(dpi=150, clip=clip).tobytes("png")
        if job and p % 20 == 0:
            job.pages_complete = p + 1
            job.progress = int((p / total) * 15)
    doc.close()

    if job:
        job.phase = "ocr"

    custs = []
    cl = threading.Lock()
    done = [0]

    def ocr(p):
        img = Image.open(io.BytesIO(imgs[p]))
        text = pytesseract.image_to_string(img, config='--psm 6')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        c = _detect_ocr(lines, p + 1)
        if c:
            with cl:
                k = f"{c['name']}|{c['state']}|{c['zip']}"
                if not any(f"{x['name']}|{x['state']}|{x['zip']}" == k for x in custs):
                    custs.append(c)
        done[0] += 1
        if job and done[0] % 10 == 0:
            job.pages_complete = done[0]
            job.customers_found = len(custs)
            job.progress = 15 + int((done[0] / total) * 75)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for f in as_completed([ex.submit(ocr, p) for p in range(total)]):
            f.result()

    custs.sort(key=lambda c: c['pageStart'])
    for i, c in enumerate(custs):
        c['pageEnd'] = custs[i + 1]['pageStart'] - 1 if i + 1 < len(custs) else total
        c['pageCount'] = c['pageEnd'] - c['pageStart'] + 1
        c['index'] = i + 1
    return custs


def _detect_ocr(lines, pnum):
    zone = lines[:max(1, int(len(lines) * 0.4))]
    for i, line in enumerate(zone):
        lc = clean_line(line.strip())
        m = CITY_STATE_ZIP_RE.match(lc)
        if not m:
            continue
        city, state, zc = m.group(1).strip(), m.group(2).upper(), m.group(3)
        if state not in US_STATES or 'ZURICH' in city.upper() or 'GREENVILLE' in city.upper():
            continue
        street = name = None
        conf = 'weak'
        al = [lc]
        if i >= 1:
            prev = clean_line(zone[i - 1].strip())
            if STREET_RE.match(prev):
                street = prev
                al.insert(0, street)
                if i >= 2:
                    nl = clean_line(zone[i - 2].strip())
                    if is_likely_person_name(nl):
                        name = nl
                        conf = 'strong'
                        if i >= 3:
                            co = clean_line(zone[i - 3].strip())
                            if is_likely_person_name(co):
                                name = f"{co} / {nl}"
                        al.insert(0, name)
                if not name:
                    conf = 'medium'
            elif is_likely_person_name(prev):
                name = prev
                conf = 'medium'
                al.insert(0, name)
        return {
            'name': name or '(Not detected)', 'street': street or '',
            'city': city, 'state': state, 'zip': zc,
            'cityStateZip': f"{city}, {state} {zc}",
            'pageStart': pnum, 'pageEnd': None,
            'confidence': conf, 'addressLines': al,
        }
    return None


# ---------------------------------------------------------------------------
# Main processor — text first, OCR fallback
# ---------------------------------------------------------------------------
def process_pdf(pdf_path, batch_size=12, job=None):
    start = time.time()
    if job:
        job.phase = "text_scan"
        job.status = "processing"

    custs = detect_boundaries_text(pdf_path, job)
    if custs and len(custs) >= 2:
        mode = 'text'
        if job:
            job.mode_used = 'text'
    else:
        mode = 'ocr'
        if job:
            job.phase = "ocr_fallback"
            job.mode_used = 'ocr'
        custs = detect_boundaries_ocr(pdf_path, job)

    elapsed = time.time() - start
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
def split_pdf_into_batches(pdf_path, batches, output_dir):
    doc = fitz.open(pdf_path)
    files = []
    for b in batches:
        bd = fitz.open()
        bd.insert_pdf(doc, from_page=b['pageStart'] - 1, to_page=b['pageEnd'] - 1)
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
def job_split_only(job, pdf_path, output_dir):
    """Background worker for /split — split PDF, create ZIP for download."""
    try:
        result = process_pdf(pdf_path, job.batch_size, job)
        bf = split_pdf_into_batches(pdf_path, result['manifest']['batches'], output_dir)

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


def job_split_and_upload(job, pdf_path, output_dir, vertesia_jwt, collection_id=None):
    """Background worker for /split-and-upload — split PDF, upload to Vertesia."""
    try:
        result = process_pdf(pdf_path, job.batch_size, job)
        bf = split_pdf_into_batches(pdf_path, result['manifest']['batches'], output_dir)

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
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, f.filename, batch_size)

    td = tempfile.mkdtemp()
    od = os.path.join(td, 'batches')
    os.makedirs(od)
    pp = os.path.join(td, 'input.pdf')
    f.save(pp)

    with jobs_lock:
        jobs[job_id] = job

    threading.Thread(target=job_split_only, args=(job, pp, od), daemon=True).start()
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
        args=(job, pp, od, vertesia_jwt, collection_id),
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"PDF Chunker on :{port}")
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', '') == 'true')
