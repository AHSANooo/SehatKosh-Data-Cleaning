#!/usr/bin/env python3
"""
batch_redact.py - High-Performance Offline PII De-identification Pipeline
for Medical Prescriptions, Clinical Notes & Receipts.

Optimized for:
  - Intel Core i5-7300HQ (4 Cores CPU multi-threading, oneDNN PIR safe mode)
  - PaddleOCR (PP-OCRv4 Mobile / PP-OCRv6 Medium)
  - spaCy + Microsoft Presidio with en_core_web_sm
  - PyMuPDF ultra-fast vector rasterization
  - Resumable batch execution with progress manifest and error tracking
"""

import os
# Disable slow online model source checks and optimize paddle allocator
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import sys
import re
import gc
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("redaction.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
logger = logging.getLogger("batch_redact")

# Global engines (lazy loaded)
ocr_engine = None
analyzer_engine = None

# Pakistani Medical Forms Regex Patterns
REGEX_PII = {
    "CNIC_FORMATTED": re.compile(r"\b\d{5}[-–]\d{7}[-–]\d\b"),
    "CNIC_RAW": re.compile(r"\b\d{13}\b"),
    "PHONE_MOBILE": re.compile(r"(?:\+92|0092|0)?3\d{2}[-\s]?\d{7}\b"),
    "PHONE_LANDLINE": re.compile(r"\b0[24589]\d{1,2}[-\s]?\d{6,7}\b"),
    "MRNO_PIN": re.compile(r"\b(?:mr|mrno|pin|reg|uhid|ipd|opd)[-–\s#.:]*\d+[-–\w/]*\b", re.IGNORECASE),
    "TOKEN_NO": re.compile(r"\b(?:token|rec(?:eipt)?)[-–\s#.:]*\d+\b", re.IGNORECASE),
    "ADMISSION_NO": re.compile(r"\b(?:adm(?:ission)?)[-–\s#.:]*\d+[-–\w]*\b", re.IGNORECASE),
}

# Anchor phrases where text in same box or adjacent horizontal boxes contains PII
ANCHOR_LABELS = [
    "name", "patient's name", "patient name", "pt name", "pt. name",
    "s/o", "d/o", "w/o", "c/o", "b/o", "f/name", "father name", "father's name",
    "father / husband name", "guardian name", "mrno", "mr no", "mr #",
    "pin", "pin #", "cnic", "cnic no", "nic", "contact", "contact no",
    "phone", "cell", "mob", "mobile", "tel", "admission no", "adm no",
    "token #", "token no", "receipt no", "bed #", "bed no", "address",
    "residence"
]

# Honorifics / Name prefixes common in Pakistani prescriptions
NAME_PREFIXES = ["mr.", "mr ", "mrs.", "mrs ", "ms.", "ms ", "miss ", "master ", "baby ", "b/o ", "s/o ", "d/o ", "w/o "]

# Medical terms and stop-words to skip in NER to prevent false positive redactions
MEDICAL_STOP_WORDS = {
    "mg", "ml", "mcg", "gm", "g", "tablet", "tab", "capsule", "cap", "syrup", "syp",
    "injection", "inj", "drops", "cream", "gel", "ointment", "sachet", "suspension",
    "bd", "tds", "od", "hs", "sos", "prn", "stat", "po", "iv", "im", "sc", "daily",
    "morning", "evening", "night", "days", "weeks", "months", "dose", "rx",
    "hospital", "clinic", "dispensary", "pharmacy", "laboratory", "lab", "department",
    "doctor", "dr", "consultant", "physician", "surgeon", "date", "signature",
    "diagnosis", "history", "examination", "investigation", "treatment", "instruction"
}


def natural_key(p: Path):
    """Natural sort key to sort 1.pdf, 2.pdf, ..., 10.pdf correctly."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.name)]


def get_bounding_rect(poly) -> Tuple[int, int, int, int]:
    """Converts 4-point polygon into (xmin, ymin, xmax, ymax)."""
    pts = np.array(poly, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return int(x), int(y), int(x + w), int(y + h)


def is_adjacent_horizontally(box1: Tuple[int, int, int, int], 
                             box2: Tuple[int, int, int, int], 
                             y_threshold: int = 30, 
                             x_max_gap: int = 350) -> bool:
    """
    Detects if box2 is directly to the right of box1 on the same text line.
    Tuned for typical receipt and prescription row layouts.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Vertical overlap / alignment check
    h1 = max(1, y1_max - y1_min)
    h2 = max(1, y2_max - y2_min)
    avg_h = (h1 + h2) / 2.0

    y_aligned = (
        abs(y1_min - y2_min) <= y_threshold or
        abs(y1_max - y2_max) <= y_threshold or
        abs((y1_min + y1_max) / 2.0 - (y2_min + y2_max) / 2.0) <= (avg_h * 0.75)
    )
    # Must be positioned to the right of box1
    x_adjacent = 0 <= (x2_min - x1_max) <= x_max_gap

    return y_aligned and x_adjacent


def init_ocr(cpu_threads: int = 4, angle_cls: bool = False, ocr_version: str = "PP-OCRv4"):
    """
    Initialize PaddleOCR with safe CPU parameters and chosen model version.
    """
    global ocr_engine
    if ocr_engine is not None:
        return ocr_engine

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.error("PaddleOCR is not installed. Run: pip install paddlepaddle paddleocr")
        raise

    logger.info(f"Initializing PaddleOCR (version={ocr_version}, CPU threads={cpu_threads})...")
    
    ocr_kwargs = {
        "device": "cpu",
        "enable_mkldnn": False,
        "cpu_threads": cpu_threads,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": angle_cls,
        "lang": "en"
    }
    
    if ocr_version:
        ocr_kwargs["ocr_version"] = ocr_version

    ocr_engine = PaddleOCR(**ocr_kwargs)
    logger.info(f"PaddleOCR {ocr_version} successfully initialized.")
    return ocr_engine


def init_ner():
    """
    Initialize Presidio AnalyzerEngine configured with high-speed en_core_web_sm.
    """
    global analyzer_engine
    if analyzer_engine is not None:
        return analyzer_engine

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        # Configure lightweight spaCy model for ultra-fast local inference (~0.2s/doc)
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        analyzer_engine = AnalyzerEngine(nlp_engine=nlp_engine)
        logger.info("Presidio AnalyzerEngine initialized with fast en_core_web_sm model.")
    except Exception as e:
        logger.warning(f"Fast Presidio config failed ({e}). Attempting default AnalyzerEngine.")
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer_engine = AnalyzerEngine()
            logger.info("Presidio AnalyzerEngine initialized with default NLP model.")
        except Exception as e2:
            logger.warning(f"Presidio AnalyzerEngine not available ({e2}). Running with Regex + Anchor matching.")
            analyzer_engine = None

    return analyzer_engine


def process_image(img_cv: np.ndarray, 
                  ocr, 
                  analyzer, 
                  pad_x: int = 4, 
                  pad_y: int = 2,
                  max_ocr_dim: int = 1400) -> Tuple[np.ndarray, int]:
    """
    Core De-identification logic for a single CV2 image.
    Returns:
      (redacted_image, total_boxes_redacted)
    """
    h_img, w_img = img_cv.shape[:2]
    
    # Scale huge images down for OCR inference while keeping output at original full resolution
    scale_factor = 1.0
    ocr_input = img_cv
    max_dim = max(h_img, w_img)
    if max_dim > max_ocr_dim:
        scale_factor = float(max_ocr_dim) / max_dim
        new_w = int(w_img * scale_factor)
        new_h = int(h_img * scale_factor)
        ocr_input = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

    try:
        predictions = ocr.predict(ocr_input)
    except Exception as e:
        logger.error(f"OCR prediction failed: {e}")
        return img_cv, 0

    if not predictions:
        return img_cv, 0

    parsed_lines = []
    pred = predictions[0]

    # Handle PaddleOCR dictionary prediction format
    if isinstance(pred, dict):
        rec_texts = pred.get("rec_texts", [])
        rec_scores = pred.get("rec_scores", [])
        rec_boxes = pred.get("rec_boxes", [])
        rec_polys = pred.get("rec_polys", [])

        for idx, text in enumerate(rec_texts):
            clean_text = str(text).strip()
            if not clean_text:
                continue
            conf = float(rec_scores[idx]) if idx < len(rec_scores) else 1.0

            if idx < len(rec_boxes):
                b = rec_boxes[idx]
                rect = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            elif idx < len(rec_polys):
                rect = get_bounding_rect(rec_polys[idx])
            else:
                continue

            if scale_factor != 1.0:
                inv = 1.0 / scale_factor
                rect = (
                    int(rect[0] * inv),
                    int(rect[1] * inv),
                    int(rect[2] * inv),
                    int(rect[3] * inv)
                )

            parsed_lines.append({
                "rect": rect,
                "text": clean_text,
                "lower": clean_text.lower(),
                "conf": conf
            })

    # Legacy PaddleOCR list format fallback
    elif isinstance(pred, list):
        for item in pred:
            poly, (text, conf) = item
            rect = get_bounding_rect(poly)
            if scale_factor != 1.0:
                inv = 1.0 / scale_factor
                rect = (
                    int(rect[0] * inv),
                    int(rect[1] * inv),
                    int(rect[2] * inv),
                    int(rect[3] * inv)
                )
            clean_text = str(text).strip()
            parsed_lines.append({
                "rect": rect,
                "text": clean_text,
                "lower": clean_text.lower(),
                "conf": conf
            })

    if not parsed_lines:
        return img_cv, 0

    # Sort parsed text lines in reading order (top-to-bottom, left-to-right)
    parsed_lines.sort(key=lambda item: (item["rect"][1] // 25, item["rect"][0]))
    num_lines = len(parsed_lines)
    boxes_to_redact: List[Tuple[int, int, int, int]] = []

    for i, curr in enumerate(parsed_lines):
        text = curr["text"]
        text_lower = curr["lower"]
        box_redacted = False

        # --- Strategy A: Deterministic Regex Patterns ---
        for p_name, pattern in REGEX_PII.items():
            if pattern.search(text):
                boxes_to_redact.append(curr["rect"])
                box_redacted = True
                break

        if box_redacted:
            continue

        # --- Strategy B: Anchor-Label Expansion ---
        for anchor in ANCHOR_LABELS:
            if anchor in text_lower:
                # Case B1: The anchor AND value are inside the SAME box (e.g. "Name: Muhammad Ali")
                pos = text_lower.find(anchor)
                remainder = text_lower[pos + len(anchor):].strip(" :-\t")
                if len(remainder) >= 2:
                    boxes_to_redact.append(curr["rect"])
                    box_redacted = True

                # Case B2: Value is in adjacent horizontal box(es)
                for j in range(i + 1, min(i + 5, num_lines)):
                    target = parsed_lines[j]
                    if is_adjacent_horizontally(curr["rect"], target["rect"]):
                        # Avoid redacting if target itself is another anchor label (e.g., "Age:")
                        target_is_label = any(l == target["lower"].rstrip(" :") for l in ANCHOR_LABELS)
                        if not target_is_label:
                            boxes_to_redact.append(target["rect"])
                break

        if box_redacted:
            continue

        # Strategy B3: Honorifics / Name prefixes (e.g. "Mr. Asad", "Baby of Fatima")
        for prefix in NAME_PREFIXES:
            if text_lower.startswith(prefix) and len(text_lower) > len(prefix) + 2:
                boxes_to_redact.append(curr["rect"])
                box_redacted = True
                break

        if box_redacted:
            continue

        # --- Strategy C: Local NER with Presidio / spaCy ---
        if analyzer and len(text) > 3:
            tokens = set(re.findall(r"\b[a-zA-Z]+\b", text_lower))
            if tokens and not tokens.issubset(MEDICAL_STOP_WORDS):
                try:
                    ner_results = analyzer.analyze(
                        text=text, 
                        entities=["PERSON", "LOCATION"], 
                        language='en'
                    )
                    if ner_results:
                        boxes_to_redact.append(curr["rect"])
                except Exception:
                    pass

    # Draw crisp black boxes with padding
    img_out = img_cv.copy()
    unique_boxes = list(set(boxes_to_redact))

    for (x1, y1, x2, y2) in unique_boxes:
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w_img, x2 + pad_x)
        y2_pad = min(h_img, y2 + pad_y)
        cv2.rectangle(img_out, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 0, 0), -1)

    return img_out, len(unique_boxes)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Loads existing progress manifest for resumable execution."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]):
    """Saves progress manifest atomically."""
    temp_path = manifest_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    temp_path.replace(manifest_path)


def run_batch_pipeline(
    input_dir: str = "./Data",
    output_dir: str = "./redacted_documents",
    dpi: int = 150,
    angle_cls: bool = False,
    ocr_version: str = "PP-OCRv4",
    max_ocr_dim: int = 1400,
    sample_limit: Optional[int] = None,
    force_reprocess: bool = False
):
    """
    Main batch processing pipeline.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return

    # Initialize Hardware Engines
    ocr = init_ocr(cpu_threads=4, angle_cls=angle_cls, ocr_version=ocr_version)
    analyzer = init_ner()

    # Discover documents with natural sorting
    supported_exts = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = sorted([f for f in input_path.glob("*") if f.suffix.lower() in supported_exts], key=natural_key)
    total_found = len(files)
    logger.info(f"Discovered {total_found} documents in {input_path}")

    if sample_limit and sample_limit > 0:
        files = files[:sample_limit]
        logger.info(f"Sample limit applied: Processing first {len(files)} files.")

    # Resumption Manifest
    manifest_path = output_path / "redaction_manifest.json"
    manifest = load_manifest(manifest_path)

    # Filter out already processed files
    if not force_reprocess:
        unprocessed = [f for f in files if f.name not in manifest or manifest[f.name].get("status") != "success"]
        logger.info(f"Resumption check: {len(files) - len(unprocessed)} already done, {len(unprocessed)} remaining.")
        files_to_process = unprocessed
    else:
        files_to_process = files

    if not files_to_process:
        logger.info("All documents have already been successfully processed. Nothing to do!")
        return

    # High-speed PDF rendering
    use_pymupdf = False
    try:
        import fitz  # PyMuPDF is ~10x faster than pdf2image
        use_pymupdf = True
        logger.info("PyMuPDF (fitz) detected: Utilizing ultra-fast direct rasterizer.")
    except ImportError:
        try:
            from pdf2image import convert_from_path
            logger.info("Using pdf2image + poppler rasterizer.")
        except ImportError:
            logger.error("Neither PyMuPDF nor pdf2image installed. Run: pip install pymupdf or pdf2image")
            return

    # Metrics
    success_count = 0
    error_count = 0
    total_redacted_boxes = 0
    start_total_time = time.time()

    progress_bar = tqdm(files_to_process, desc="De-identifying Documents", unit="doc")

    for file_idx, file in enumerate(progress_bar):
        file_start = time.time()
        file_boxes = 0
        pages_processed = 0

        try:
            if file.suffix.lower() == ".pdf":
                if use_pymupdf:
                    # PyMuPDF fast rasterization
                    doc = fitz.open(str(file))
                    pages_processed = len(doc)
                    for page_num in range(pages_processed):
                        page = doc.load_page(page_num)
                        # Render at desired DPI (72 is default resolution in PDF)
                        zoom = dpi / 72.0
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        
                        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, 3))
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                        redacted, boxes = process_image(img_cv, ocr, analyzer, max_ocr_dim=max_ocr_dim)
                        file_boxes += boxes

                        if pages_processed == 1:
                            out_filename = output_path / f"{file.stem}.png"
                        else:
                            out_filename = output_path / f"{file.stem}_page{page_num + 1}.png"
                        cv2.imwrite(str(out_filename), redacted)
                    doc.close()

                else:
                    # pdf2image fallback
                    pages = convert_from_path(str(file), dpi=dpi, thread_count=4)
                    pages_processed = len(pages)
                    for idx, page in enumerate(pages):
                        img_np = np.array(page)
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        redacted, boxes = process_image(img_cv, ocr, analyzer, max_ocr_dim=max_ocr_dim)
                        file_boxes += boxes

                        if pages_processed == 1:
                            out_filename = output_path / f"{file.stem}.png"
                        else:
                            out_filename = output_path / f"{file.stem}_page{idx + 1}.png"
                        cv2.imwrite(str(out_filename), redacted)

            else:
                # Direct image file
                img_cv = cv2.imread(str(file))
                if img_cv is None:
                    raise ValueError("Failed to read image file with cv2.imread")
                pages_processed = 1
                redacted, boxes = process_image(img_cv, ocr, analyzer, max_ocr_dim=max_ocr_dim)
                file_boxes += boxes
                out_filename = output_path / f"{file.stem}.png"
                cv2.imwrite(str(out_filename), redacted)

            elapsed = time.time() - file_start
            success_count += 1
            total_redacted_boxes += file_boxes

            manifest[file.name] = {
                "status": "success",
                "pages": pages_processed,
                "boxes_redacted": file_boxes,
                "duration_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

        except Exception as e:
            error_count += 1
            elapsed = time.time() - file_start
            logger.error(f"Error processing {file.name}: {e}")
            manifest[file.name] = {
                "status": "error",
                "error": str(e),
                "duration_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

        # Periodic checkpointing and memory garbage collection
        if (file_idx + 1) % 25 == 0:
            save_manifest(manifest_path, manifest)
            gc.collect()

        progress_bar.set_postfix({
            "Done": success_count,
            "Errors": error_count,
            "PII Boxes": total_redacted_boxes,
            "Last(s)": f"{elapsed:.1f}"
        })

    # Final save
    save_manifest(manifest_path, manifest)

    total_time = time.time() - start_total_time
    logger.info("=" * 60)
    logger.info("BATCH REDACTION COMPLETED")
    logger.info(f"Processed: {success_count} | Errors: {error_count} | Total PII Redacted: {total_redacted_boxes}")
    logger.info(f"Total Time: {total_time:.1f}s ({total_time / 60:.2f} mins)")
    if success_count > 0:
        logger.info(f"Average Speed: {total_time / success_count:.2f} s/doc")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Manifest saved to: {manifest_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance Medical Document PII De-identification Pipeline"
    )
    parser.add_argument("--input_dir", type=str, default="./Data", help="Path to input directory containing PDFs/Images (default: ./Data)")
    parser.add_argument("--output_dir", type=str, default="./redacted_documents", help="Path to output directory (default: ./redacted_documents)")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI (default: 150 for optimal balance of speed and clarity)")
    parser.add_argument("--max_ocr_dim", type=int, default=1400, help="Max dimension for OCR scaling (default: 1400px)")
    parser.add_argument("--ocr_version", type=str, default="PP-OCRv4", choices=["PP-OCRv4", "PP-OCRv6"], help="PaddleOCR model version (default: PP-OCRv4)")
    parser.add_argument("--angle_cls", action="store_true", help="Enable text angle classification (slower, but fixes upside-down/rotated text)")
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N files for testing/benchmarking")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already completed files")

    args = parser.parse_args()

    run_batch_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        angle_cls=args.angle_cls,
        ocr_version=args.ocr_version,
        max_ocr_dim=args.max_ocr_dim,
        sample_limit=args.sample,
        force_reprocess=args.force
    )


if __name__ == "__main__":
    main()
