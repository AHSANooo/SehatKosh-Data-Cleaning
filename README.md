# SehatKosh Medical Document De-Identification Pipeline

High-performance, offline, and air-gapped PII de-identification pipeline designed for medical prescriptions, receipt vouchers, and clinical notes. Built specifically for **WSL2 (Ubuntu 24.04)** on **Windows 10**, optimized for the **Intel Core i5-7300HQ (4 Cores)** and **NVIDIA GeForce GTX 1050 (4 GB VRAM)**.

---

## ⚡ Architecture & Key Optimizations

1. **Ultra-Fast PDF Rasterization (`PyMuPDF`)**: Direct C++ bindings render PDF pages ~10x faster than traditional `pdf2image` + `pdftoppm`.
2. **Local Multi-Threaded OCR (`PaddleOCR PP-OCRv4 / PP-OCRv6`)**:
   - `PP-OCRv4` Mobile models deliver rapid ~5–10s/doc processing on CPU with full bounding box extraction.
   - Proportional resolution scaling preserves original document quality while maintaining high OCR throughput.
   - Configured with `enable_mkldnn=False` to ensure stability across oneDNN / PIR graph backends.
3. **Multi-Strategy PII Matcher**:
   - **Strategy A (Deterministic Regex)**: Captures Pakistani CNICs (formatted `XXXXX-XXXXXXX-X` and 13-digit raw), Pakistani mobile numbers (`03XX-XXXXXXX`, `+92`), MRNOs, Registration IDs, Token numbers, and Admission numbers.
   - **Strategy B (Anchor-Label Expansion)**: Detects anchor phrases (`Patient Name`, `Father/Husband Name`, `S/O`, `D/O`, `W/O`, `CNIC`, `Contact`, `Bed #`, etc.) and automatically redacts either inline text or adjacent horizontal bounding boxes.
   - **Strategy B3 (Honorifics & Prefixes)**: Redacts patient names starting with `Mr.`, `Mrs.`, `Ms.`, `Miss`, `Master`, `Baby`, etc.
   - **Strategy C (Local NLP NER)**: High-speed `spaCy` + `Microsoft Presidio` analyzer loaded with `en_core_web_sm` (~0.2s/doc) detects names and geographic locations while ignoring medical terminology stop-words (`mg`, `ml`, `tab`, `iv`, `rx`, etc.).
4. **Crisp Black-Box Masking (`OpenCV`)**: Adds customizable pixel padding around identified bounding boxes and fills them with solid black.
5. **Stateful Manifest & Crash Recovery (`redaction_manifest.json`)**: Every processed document is logged atomically. If interrupted, restarting the script will automatically resume from where it stopped.

---

## 📁 Project Structure

```
"SehatKosh Data Cleaning"/
├── Data/                       # Input folder containing 2,717 raw PDF documents
├── redacted_documents/         # Output folder for de-identified images and manifests
│   ├── 1.png
│   ├── 2.png
│   ├── ...
│   ├── redaction_manifest.json # State tracker (status, pages, boxes, processing time)
│   └── audit_review.csv        # Exported human-in-the-loop audit records
├── redaction_env/              # Dedicated Python 3.12 virtual environment (WSL2)
├── batch_redact.py             # Main batch de-identification pipeline
├── streamlit_audit.py          # Interactive side-by-side human audit web app
├── run_redaction.sh            # WSL2 / Linux launcher for batch redaction
├── run_audit.sh                # WSL2 / Linux launcher for Streamlit audit UI
├── run_redaction.bat           # 1-Click Windows shortcut for batch redaction
├── run_audit.bat               # 1-Click Windows shortcut for Streamlit audit UI
├── redaction.log               # Detailed log file
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start Instructions

### 1. Run De-Identification (Full Batch)

#### Option A: 1-Click from Windows
Double-click [`run_redaction.bat`](file:///d:/Projects/SehatKosh%20Data%20Cleaning/run_redaction.bat).

#### Option B: From WSL2 Terminal
```bash
wsl -d Ubuntu-24.04 bash -c "cd '/mnt/d/Projects/SehatKosh Data Cleaning' && ./run_redaction.sh"
```

---

### 2. Run Human-in-the-Loop Audit UI

To spot-check redacted documents side-by-side with the original unredacted files:

#### Option A: 1-Click from Windows
Double-click [`run_audit.bat`](file:///d:/Projects/SehatKosh%20Data%20Cleaning/run_audit.bat).

#### Option B: From WSL2 Terminal
```bash
wsl -d Ubuntu-24.04 bash -c "cd '/mnt/d/Projects/SehatKosh Data Cleaning' && ./run_audit.sh"
```
Then open your browser to: **[http://localhost:8501](http://localhost:8501)**

**Audit Features:**
- **Side-by-Side View**: Compares the raw document against the de-identified output.
- **One-Click Actions**: `Approve & Next` (Green), `Flag & Next` (Red with optional notes).
- **Filters**: Quickly filter by `Only Flagged` or `Only Pending`.
- **Export**: Download `audit_review.csv` directly from the sidebar.

---

## ⚙️ Advanced CLI Options

You can pass arguments to [`batch_redact.py`](file:///d:/Projects/SehatKosh%20Data%20Cleaning/batch_redact.py) directly:

```bash
# Run a quick test on 10 documents
python3 batch_redact.py --sample 10

# Force re-processing of already completed files
python3 batch_redact.py --force

# Adjust rasterization DPI (default: 150)
python3 batch_redact.py --dpi 200

# Switch to PP-OCRv6 medium models (slower, maximum accuracy)
python3 batch_redact.py --ocr_version PP-OCRv6

# Enable textline orientation angle correction (for skewed or rotated scans)
python3 batch_redact.py --angle_cls
```

---

## 🔒 Privacy & Offline Guarantee

- **Air-Gapped Execution**: All OCR inference and NLP Named Entity Recognition run 100% locally on your machine.
- **Zero API Fees**: No external cloud APIs (Google Vision, AWS Textract, or Azure) are used.
- **Data Integrity**: Input documents inside `Data/` are read-only and never modified or overwritten.

