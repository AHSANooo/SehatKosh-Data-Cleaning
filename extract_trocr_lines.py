#!/usr/bin/env python3
"""
extract_trocr_lines.py - Step 1: Automated Page Orientation Fix & Handwritten Line Extraction.

Processes cleaned and de-identified medical document pages:
  1. Detects and corrects page orientation (0°, 90°, 180°, 270°).
  2. Reuses OCR detections when orientation is already upright to eliminate redundant computation.
  3. Proportional inference scaling for high throughput while extracting full-resolution crops.
  4. Filters out static printed hospital form headers and field labels.
  5. Merges fragmented handwritten lines horizontally without crossing printed form labels.
  6. Applies anti-clipping padding (+14px horizontal, +10px vertical) to protect ascenders and descenders.
  7. Generates full-page verification overlays with numbered red boxes (doc_id_boxed.png).
  8. Saves line crops (crops/{doc_id}_L001.png) and Hugging Face ready metadata.csv & annotation_sheet.csv.
  9. Resumable execution via atomic extraction_manifest.json.
"""

import os
# Optimize paddle inference and disable slow online source checks
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import sys
import re
import gc
import csv
import json
import time
import argparse
import logging
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
        logging.FileHandler("extraction.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("extract_trocr_lines")

# Global OCR engine
ocr_engine = None

# Known printed form phrases to exclude from handwritten line crops
PRINTED_FORM_PHRASES = [
    # Hospital names & headers
    "alkhidmat", "raazi hospital", "cbr town", "phase-i", "phase-1", "islamabad",
    "operation theater notes", "daily consultant notes", "treatment ticket",
    "emergency ticket", "prescription", "discharge summary", "clinical notes",
    # Form labels & headers
    "patient name", "patient's name", "pt name", "pt. name",
    "father/husband name", "father / husband name", "father name", "husband name", "f/name",
    "gender", "sex", "age", "age/gender", "age / gender", "age / sex",
    "admission no", "adm no", "adm. no", "reg no", "reg. no", "mrno", "mr no", "pin", "pin #",
    "surgeon name", "anaesthesiologist name", "anesthesiologist name",
    "date & time", "date", "time", "procedure", "operation notes",
    "post operative orders", "signature", "days and date", "drug",
    "dose & route", "interval", "mo signature", "nurse", "shift", "initial",
    "hospital receipt", "token #", "token no", "receipt no", "cnic", "contact",
    "panel", "address", "residence", "service", "rate", "price", "total",
    "received", "balance", "integcubes", "doctor", "dr", "room no", "bed #",
    "bed no", "ref by", "visit date", "dept", "department", "unit", "consultant",
    "sub total", "net total", "amount", "discount", "qty", "quantity", "cashier",
    "history", "examination", "investigation", "diagnosis", "rx", "treatment"
]

UPRIGHT_ANCHORS = {
    "alkhidmat", "hospital", "notes", "patient", "surgeon", "orders",
    "ticket", "receipt", "gender", "procedure", "consultant", "signature",
    "doctor", "treatment", "medicine", "table", "rate", "total", "date",
    "raazi", "islamabad", "name", "address", "phone"
}


def natural_key(p: Path):
    """Natural sort key to sort 1.png, 2.png, ..., 10.png correctly."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.name)]


def poly_to_rect(poly) -> List[int]:
    """Converts a 4-point polygon into [xmin, ymin, xmax, ymax]."""
    pts = np.array(poly, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return [int(x), int(y), int(x + w), int(y + h)]


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotates an image by 90, 180, or 270 degrees clockwise."""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def init_ocr(device: str = "auto", cpu_threads: int = 4, ocr_version: str = "PP-OCRv4"):
    """
    Initializes PaddleOCR with optimal speed/accuracy settings.
    Default: PP-OCRv4 Mobile for fast local inference.
    """
    global ocr_engine
    if ocr_engine is not None:
        return ocr_engine

    from paddleocr import PaddleOCR
    import paddle

    use_gpu = False
    if device.lower() in ["auto", "gpu"]:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            use_gpu = True

    chosen_device = "gpu:0" if use_gpu else "cpu"
    logger.info(f"Initializing PaddleOCR (version={ocr_version}, device={chosen_device}, cpu_threads={cpu_threads})...")

    ocr_kwargs = {
        "device": chosen_device,
        "enable_mkldnn": False,
        "cpu_threads": cpu_threads,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "text_det_limit_side_len": 960,
        "lang": "en"
    }

    if ocr_version:
        ocr_kwargs["ocr_version"] = ocr_version

    try:
        ocr_engine = PaddleOCR(**ocr_kwargs)
        logger.info(f"PaddleOCR {ocr_version} successfully initialized on {chosen_device}.")
    except Exception as e:
        logger.warning(f"PaddleOCR failed with {chosen_device} ({e}). Falling back to CPU.")
        ocr_kwargs["device"] = "cpu"
        ocr_engine = PaddleOCR(**ocr_kwargs)

    return ocr_engine


def run_ocr_inference(ocr, img: np.ndarray, max_dim: int = 1500) -> List[Tuple[Any, Tuple[str, float]]]:
    """
    Runs PaddleOCR on image with optional proportional scaling for speed,
    rescaling bounding box coordinates back to full-resolution dimensions.
    """
    h_img, w_img = img_cv_shape = img.shape[:2]
    scale_factor = 1.0
    ocr_input = img

    curr_max = max(h_img, w_img)
    if curr_max > max_dim:
        scale_factor = float(max_dim) / curr_max
        new_w = int(w_img * scale_factor)
        new_h = int(h_img * scale_factor)
        ocr_input = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    try:
        res = ocr.predict(ocr_input)
    except Exception:
        try:
            res = ocr.ocr(ocr_input, cls=True)
        except Exception as e:
            logger.error(f"OCR inference failed: {e}")
            return []

    if not res:
        return []

    items = []
    first = res[0]
    inv_scale = 1.0 / scale_factor if scale_factor != 1.0 else 1.0

    # PaddleOCR 3.7+ dictionary prediction format
    if isinstance(first, dict):
        rec_texts = first.get("rec_texts", [])
        rec_scores = first.get("rec_scores", [])
        rec_polys = first.get("rec_polys", [])
        rec_boxes = first.get("rec_boxes", [])

        for idx, text in enumerate(rec_texts):
            clean_text = str(text).strip()
            if not clean_text:
                continue
            conf = float(rec_scores[idx]) if idx < len(rec_scores) else 1.0

            if idx < len(rec_polys):
                poly = rec_polys[idx]
                if scale_factor != 1.0:
                    poly = (np.array(poly) * inv_scale).astype(np.int32)
            elif idx < len(rec_boxes):
                b = rec_boxes[idx]
                x1, y1, x2, y2 = int(b[0] * inv_scale), int(b[1] * inv_scale), int(b[2] * inv_scale), int(b[3] * inv_scale)
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            else:
                continue

            items.append((poly, (clean_text, conf)))

    # Legacy list format [[poly, (text, conf)], ...]
    elif isinstance(first, list):
        for entry in first:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                poly, text_conf = entry[0], entry[1]
                if scale_factor != 1.0:
                    poly = (np.array(poly) * inv_scale).astype(np.int32)
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    items.append((poly, (str(text_conf[0]).strip(), float(text_conf[1]))))

    return items


def fix_page_orientation(img: np.ndarray, ocr) -> Tuple[np.ndarray, int, List[Tuple[Any, Tuple[str, float]]]]:
    """
    Detects 90/180/270 degree page rotation.
    Returns: (upright_img, rotated_deg, lines_on_upright_img).
    Crucial optimization: If already upright (0°), returns the already-computed lines,
    cutting processing time by 50%!
    """
    lines = run_ocr_inference(ocr, img)
    if not lines:
        return img, 0, []

    vertical_boxes = 0
    horizontal_boxes = 0

    for poly, (text, conf) in lines:
        x1, y1, x2, y2 = poly_to_rect(poly)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        if h > w * 1.3 and len(text.strip()) >= 3:
            vertical_boxes += 1
        elif w > h * 1.3 and len(text.strip()) >= 3:
            horizontal_boxes += 1

    # Step 1: Detect sideways scans (90 or 270 degrees)
    if vertical_boxes > horizontal_boxes:
        best_angle = 0
        best_score = -1.0
        best_lines = []
        for cand_angle in [90, 270]:
            cand_img = rotate_image(img, cand_angle)
            cand_lines = run_ocr_inference(ocr, cand_img)
            score = 0.0
            for _, (txt, conf) in cand_lines:
                words = re.sub(r"[^a-zA-Z\s]", " ", txt.lower()).split()
                score += sum(2.0 for w in words if w in UPRIGHT_ANCHORS)
                score += conf * 0.2
            if score > best_score:
                best_score = score
                best_angle = cand_angle
                best_lines = cand_lines
        if best_angle != 0:
            return rotate_image(img, best_angle), best_angle, best_lines

    # Step 2: Detect upside-down scans (180 degrees)
    upright_score = 0
    for _, (txt, conf) in lines:
        words = re.sub(r"[^a-zA-Z\s]", " ", txt.lower()).split()
        upright_score += sum(3 for w in words if w in UPRIGHT_ANCHORS)

    if upright_score < 4:
        img_180 = rotate_image(img, 180)
        lines_180 = run_ocr_inference(ocr, img_180)
        score_180 = 0
        for _, (txt, conf) in lines_180:
            words = re.sub(r"[^a-zA-Z\s]", " ", txt.lower()).split()
            score_180 += sum(3 for w in words if w in UPRIGHT_ANCHORS)

        if score_180 > upright_score:
            return img_180, 180, lines_180

    # Already upright! Reuse already computed lines
    return img, 0, lines


def is_printed_form_label(img_crop: np.ndarray, text: str) -> bool:
    """
    Distinguishes static printed form headers/labels from clinical handwritten lines.
    Multi-signal: matches keyword dictionary and boilerplate, while preserving blue and black handwriting.
    """
    clean = re.sub(r"[^a-zA-Z0-9\s/&\-:#]", "", text).strip().lower()
    if not clean:
        return True

    # 1. Exact or close substring match against known printed hospital form templates
    for phrase in PRINTED_FORM_PHRASES:
        if clean == phrase:
            return True
        if len(phrase) >= 6 and phrase in clean and len(clean) <= len(phrase) + 5:
            return True
        if len(clean) >= 6 and clean in phrase:
            return True

    # 2. Hospital contact / address boilerplate
    if "2374911" in clean or "cbr town" in clean or "phase-1" in clean or "phase-i" in clean:
        return True

    # 3. Pen ink color boost: if the crop contains clear blue/purple pen ink, it is definitely handwritten
    if img_crop is not None and img_crop.size > 0:
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([85, 30, 40]), np.array([145, 255, 240]))
        ink_ratio = np.count_nonzero(blue_mask) / float(img_crop.shape[0] * img_crop.shape[1])
        if ink_ratio > 0.015:
            return False

    return False


def boxes_vertically_aligned(b1: List[int], b2: List[int]) -> bool:
    """Checks if two bounding boxes are on the same horizontal line with sufficient vertical overlap."""
    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    min_h = max(1, min(b1[3] - b1[1], b2[3] - b2[1]))
    return (y_overlap / float(min_h)) >= 0.45


def printed_label_between(b1: List[int], b2: List[int], printed_boxes: List[List[int]]) -> bool:
    """
    Prevents merging two separate form fields on the same row (e.g. 'Male' and '62 yr')
    when a printed label (e.g. 'Age') sits between them.
    """
    left_x = min(b1[2], b2[2])
    right_x = max(b1[0], b2[0])
    mid_y = (b1[1] + b1[3] + b2[1] + b2[3]) / 4.0

    for pb in printed_boxes:
        pb_cx = (pb[0] + pb[2]) / 2.0
        if left_x <= pb_cx <= right_x and pb[1] <= mid_y <= pb[3]:
            return True
    return False


def merge_same_line_boxes(hw_items: List[Dict[str, Any]], 
                          printed_boxes: List[List[int]], 
                          max_x_gap: int = 140) -> List[Dict[str, Any]]:
    """
    Merges fragmented pieces of a single handwritten line in reading order
    without merging across printed form labels.
    """
    if not hw_items:
        return []

    # Sort top-to-bottom, then left-to-right
    hw_items.sort(key=lambda item: (item["rect"][1] // 20, item["rect"][0]))

    merged: List[Dict[str, Any]] = []
    for item in hw_items:
        if not merged:
            merged.append(item)
            continue

        prev = merged[-1]
        b1 = prev["rect"]
        b2 = item["rect"]
        x_gap = b2[0] - b1[2]

        if (
            boxes_vertically_aligned(b1, b2)
            and -15 <= x_gap <= max_x_gap
            and not printed_label_between(b1, b2, printed_boxes)
        ):
            prev["rect"] = [
                min(b1[0], b2[0]),
                min(b1[1], b2[1]),
                max(b1[2], b2[2]),
                max(b1[3], b2[3]),
            ]
            prev["text"] = f"{prev['text']} {item['text']}".strip()
        else:
            merged.append(item)

    # Final reading-order sort
    merged.sort(key=lambda item: (item["rect"][1] // 18, item["rect"][0]))
    return merged


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Loads extraction manifest for crash-safe resumability."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]):
    """Saves extraction manifest atomically."""
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)


def run_pipeline(
    input_dir: str = "./redacted_documents_clean_reordered",
    output_dir: str = "./trocr_dataset",
    pad_x: int = 14,
    pad_y: int = 10,
    device: str = "auto",
    ocr_version: str = "PP-OCRv4",
    sample_limit: Optional[int] = None,
    force_reprocess: bool = False
):
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()

    upright_dir = out_path / "upright_pages"
    annotated_dir = out_path / "annotated_pages"
    crops_dir = out_path / "crops"

    for d in [upright_dir, annotated_dir, crops_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        logger.error(f"Input directory does not exist: {in_path}")
        return

    # Initialize OCR
    ocr = init_ocr(device=device, cpu_threads=4, ocr_version=ocr_version)

    # Discover documents with natural sorting
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    files = sorted([f for f in in_path.glob("*") if f.suffix.lower() in supported_exts], key=natural_key)
    logger.info(f"Discovered {len(files)} total images in {in_path}")

    if sample_limit and sample_limit > 0:
        files = files[:sample_limit]
        logger.info(f"Sample limit applied: Processing first {len(files)} files.")

    # Resumption manifest
    manifest_path = out_path / "extraction_manifest.json"
    manifest = load_manifest(manifest_path)

    # CSV Paths
    metadata_csv_path = out_path / "metadata.csv"
    annotation_sheet_path = out_path / "annotation_sheet.csv"

    # Load existing CSV rows if resuming
    existing_crop_names = set()
    metadata_rows = []
    annotation_rows = []

    if not force_reprocess and metadata_csv_path.exists():
        try:
            with open(metadata_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row:
                        metadata_rows.append(row)
                        existing_crop_names.add(Path(row[0]).name)
        except Exception:
            pass

    if not force_reprocess and annotation_sheet_path.exists():
        try:
            with open(annotation_sheet_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row:
                        annotation_rows.append(row)
        except Exception:
            pass

    # Filter files
    if not force_reprocess:
        unprocessed = [f for f in files if f.name not in manifest or manifest[f.name].get("status") != "success"]
        logger.info(f"Resumption check: {len(files) - len(unprocessed)} already done, {len(unprocessed)} remaining.")
        files_to_process = unprocessed
    else:
        files_to_process = files

    if not files_to_process:
        logger.info("All documents have already been processed! Nothing to do.")
        return

    total_crops_generated = len(existing_crop_names)
    start_time = time.time()

    progress_bar = tqdm(files_to_process, desc="Segmenting Handwritten Lines", unit="page")

    for file_idx, file_path in enumerate(progress_bar):
        doc_id = file_path.stem
        file_start = time.time()
        crops_in_doc = 0

        try:
            raw_img = cv2.imread(str(file_path))
            if raw_img is None:
                raise ValueError(f"Could not read image: {file_path}")

            # 1. Orientation fix + reuse OCR detections
            img, rotated_deg, lines = fix_page_orientation(raw_img, ocr)
            h_img, w_img = img.shape[:2]
            cv2.imwrite(str(upright_dir / f"{doc_id}.png"), img)

            hw_items = []
            printed_boxes = []

            for poly, (text, conf) in lines:
                rect = poly_to_rect(poly)
                x1, y1, x2, y2 = rect
                if (x2 - x1) < 12 or (y2 - y1) < 8:
                    continue

                sub_crop = img[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                if is_printed_form_label(sub_crop, text):
                    printed_boxes.append(rect)
                else:
                    hw_items.append({"rect": rect, "text": text.strip()})

            # 2. Horizontal layout-aware merging
            line_boxes = merge_same_line_boxes(hw_items, printed_boxes)

            # 3. Generate line crops and verification overlay
            annotated_img = img.copy()

            for line_idx, item in enumerate(line_boxes, start=1):
                x1, y1, x2, y2 = item["rect"]
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w_img, x2 + pad_x)
                cy2 = min(h_img, y2 + pad_y)

                crop = img[cy1:cy2, cx1:cx2]
                crop_name = f"{doc_id}_L{line_idx:03d}.png"
                cv2.imwrite(str(crops_dir / crop_name), crop)
                crops_in_doc += 1
                total_crops_generated += 1

                # Numbered red box overlay on QA image
                cv2.rectangle(annotated_img, (cx1, cy1), (cx2, cy2), (0, 0, 220), 2)
                cv2.putText(
                    annotated_img,
                    str(line_idx),
                    (cx1 + 4, max(20, cy1 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 220),
                    2,
                    cv2.LINE_AA,
                )

                draft_text = item["text"]

                metadata_rows.append([
                    f"crops/{crop_name}",
                    "",  # Ground-truth text filled by annotators in Step 3
                    f"{doc_id}.png",
                    line_idx,
                    rotated_deg,
                    f"{cx1},{cy1},{cx2},{cy2}",
                ])

                annotation_rows.append([
                    crop_name,
                    f"{doc_id}.png",
                    line_idx,
                    draft_text,
                    "",  # Verified text filled in Step 3
                ])

            cv2.imwrite(str(annotated_dir / f"{doc_id}_boxed.png"), annotated_img)

            elapsed = time.time() - file_start
            manifest[file_path.name] = {
                "status": "success",
                "rotated_deg": rotated_deg,
                "crops_count": crops_in_doc,
                "duration_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

        except Exception as e:
            elapsed = time.time() - file_start
            logger.error(f"Error processing {file_path.name}: {e}")
            manifest[file_path.name] = {
                "status": "error",
                "error": str(e),
                "duration_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

        # Periodic checkpoint saving every 20 documents
        if (file_idx + 1) % 20 == 0:
            save_manifest(manifest_path, manifest)
            # Write metadata.csv
            with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file_name", "text", "source_doc", "line_num", "rotated_deg", "bbox"])
                writer.writerows(metadata_rows)
            # Write annotation_sheet.csv
            with open(annotation_sheet_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["crop_file", "source_doc", "line_num", "draft_ocr", "verified_text"])
                writer.writerows(annotation_rows)
            gc.collect()

        progress_bar.set_postfix({"Total Crops": total_crops_generated, "Last(s)": f"{elapsed:.1f}"})

    # Final saves
    save_manifest(manifest_path, manifest)

    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text", "source_doc", "line_num", "rotated_deg", "bbox"])
        writer.writerows(metadata_rows)

    with open(annotation_sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_file", "source_doc", "line_num", "draft_ocr", "verified_text"])
        writer.writerows(annotation_rows)

    total_time = time.time() - start_time
    logger.info("=" * 65)
    logger.info("STEP 1: LINE EXTRACTION COMPLETED")
    logger.info(f"Processed: {len(files_to_process)} pages")
    logger.info(f"Total handwritten line crops: {total_crops_generated}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.2f} mins)")
    logger.info(f"Output directory: {out_path}")
    logger.info(f"  - Crops: {crops_dir}")
    logger.info(f"  - QA Overlays: {annotated_dir}")
    logger.info(f"  - Upright Pages: {upright_dir}")
    logger.info(f"  - Annotation Sheet: {annotation_sheet_path}")
    logger.info(f"  - HF Metadata: {metadata_csv_path}")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Automated Page Orientation Fix & Handwritten Line Extraction"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./redacted_documents_clean_reordered",
        help="Input folder of cleaned document pages (default: ./redacted_documents_clean_reordered)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trocr_dataset",
        help="Output dataset folder (default: ./trocr_dataset)"
    )
    parser.add_argument("--pad_x", type=int, default=14, help="Horizontal padding around line crops in px (default: 14)")
    parser.add_argument("--pad_y", type=int, default=10, help="Vertical padding around line crops in px (default: 10)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Compute device (default: auto)")
    parser.add_argument("--ocr_version", type=str, default="PP-OCRv4", choices=["PP-OCRv4", "PP-OCRv6"], help="PaddleOCR model version (default: PP-OCRv4)")
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N documents for testing")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already processed documents")

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        device=args.device,
        ocr_version=args.ocr_version,
        sample_limit=args.sample,
        force_reprocess=args.force
    )


if __name__ == "__main__":
    main()
