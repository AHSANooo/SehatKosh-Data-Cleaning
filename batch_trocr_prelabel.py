#!/usr/bin/env python3
"""
batch_trocr_prelabel.py - Step 2: Batched PyTorch TrOCR Draft Pre-Labeling.

Loads the extracted handwritten line crops generated in Step 1, runs local
Hugging Face TrOCR inference on GPU/CPU in memory-efficient batches, and
updates 'draft_ocr' in annotation_sheet.csv.
"""

import os
import sys
import csv
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("prelabel.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("batch_trocr_prelabel")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Batched PyTorch TrOCR Draft Pre-Labeling for Medical Handwriting"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./trocr_dataset",
        help="Path to trocr_dataset folder containing annotation_sheet.csv and crops/ (default: ./trocr_dataset)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "large", "small"],
        help="TrOCR model architecture: base (microsoft/trocr-base-handwritten), large, or small (default: base)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for GPU inference (default: 8; use 4-8 for 4GB GTX 1050, 16 for 8GB+ GPUs)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device: auto, cuda, or cpu (default: auto)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 half-precision on GPU to save VRAM and increase throughput (default: True)"
    )
    parser.add_argument(
        "--no_fp16",
        action="store_false",
        dest="fp16",
        help="Disable FP16 and run in FP32 precision"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N crops for quick benchmarking"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-inference on all crops even if draft_ocr is already populated"
    )
    return parser.parse_args()


def get_model_identifier(model_type: str) -> str:
    mapping = {
        "base": "microsoft/trocr-base-handwritten",
        "large": "microsoft/trocr-large-handwritten",
        "small": "microsoft/trocr-small-handwritten",
    }
    return mapping.get(model_type.lower(), "microsoft/trocr-base-handwritten")


def load_annotation_sheet(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Loads annotation_sheet.csv into field names and row dictionaries."""
    rows = []
    header = ["crop_file", "source_doc", "line_num", "draft_ocr", "verified_text"]
    if not csv_path.exists():
        logger.error(f"Annotation sheet not found: {csv_path}")
        return header, []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            header = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    return header, rows


def save_annotation_sheet(csv_path: Path, header: List[str], rows: List[Dict[str, str]]):
    """Saves updated annotation_sheet.csv atomically."""
    tmp_path = csv_path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def run_prelabeling(args):
    dataset_dir = Path(args.dataset_dir).resolve()
    csv_path = dataset_dir / "annotation_sheet.csv"
    crops_dir = dataset_dir / "crops"

    if not csv_path.exists():
        logger.error(f"Cannot find annotation sheet at {csv_path}. Run Step 1 (extract_trocr_lines.py) first.")
        return

    if not crops_dir.exists():
        logger.error(f"Cannot find crops directory at {crops_dir}.")
        return

    header, rows = load_annotation_sheet(csv_path)
    total_rows = len(rows)
    logger.info(f"Loaded {total_rows} total line crops from {csv_path}")

    # Determine which rows need TrOCR inference
    to_process_indices = []
    for idx, row in enumerate(rows):
        # If force is true, process all
        # If force is false, process those that have not been human-verified and have empty or Paddle-only text
        if args.force:
            to_process_indices.append(idx)
        else:
            # Check if verified_text is empty and crop exists
            if not row.get("verified_text", "").strip():
                to_process_indices.append(idx)

    if args.sample and args.sample > 0:
        to_process_indices = to_process_indices[:args.sample]
        logger.info(f"Sample limit applied: Processing {len(to_process_indices)} crops.")

    logger.info(f"Crops queued for TrOCR inference: {len(to_process_indices)}")
    if not to_process_indices:
        logger.info("All crops already have draft annotations or verified labels! Nothing to do.")
        return

    # Initialize PyTorch and TrOCR
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        logger.error("PyTorch and Hugging Face Transformers are required for Step 2.")
        logger.error("To install, run: pip install torch torchvision transformers")
        sys.exit(1)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        target_device = torch.device("cuda")
        logger.info(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
    else:
        target_device = torch.device("cpu")
        logger.info("Using CPU for inference.")

    model_name = get_model_identifier(args.model)
    logger.info(f"Loading TrOCR model ({model_name})...")
    load_start = time.time()

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    if args.fp16 and target_device.type == "cuda":
        model = model.half()
        logger.info("Enabled FP16 half-precision mode on GPU.")

    model.to(target_device)
    model.eval()
    logger.info(f"Model loaded in {time.time() - load_start:.2f}s.")

    # Batch processing loop
    batch_size = args.batch_size
    num_batches = (len(to_process_indices) + batch_size - 1) // batch_size
    start_inference_time = time.time()
    processed_count = 0

    progress_bar = tqdm(range(num_batches), desc="Running TrOCR Pre-Labeling", unit="batch")

    for b_idx in progress_bar:
        batch_slice = to_process_indices[b_idx * batch_size: (b_idx + 1) * batch_size]
        batch_images = []
        valid_indices = []

        for idx in batch_slice:
            crop_filename = rows[idx]["crop_file"]
            crop_path = crops_dir / crop_filename
            if not crop_path.exists():
                continue

            try:
                pil_img = Image.open(crop_path).convert("RGB")
                batch_images.append(pil_img)
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Could not load crop {crop_path}: {e}")

        if not batch_images:
            continue

        try:
            # Process batch
            pixel_values = processor(images=batch_images, return_tensors="pt").pixel_values
            if args.fp16 and target_device.type == "cuda":
                pixel_values = pixel_values.half()
            pixel_values = pixel_values.to(target_device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=64)

            decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, pred_text in zip(valid_indices, decoded_texts):
                rows[idx]["draft_ocr"] = pred_text.strip()
                processed_count += 1

        except Exception as e:
            logger.error(f"Inference error on batch {b_idx}: {e}")

        # Checkpoint save every 10 batches (80 crops)
        if (b_idx + 1) % 10 == 0:
            save_annotation_sheet(csv_path, header, rows)

        progress_bar.set_postfix({"Crops Updated": processed_count})

    # Final save
    save_annotation_sheet(csv_path, header, rows)

    total_time = time.time() - start_inference_time
    logger.info("=" * 65)
    logger.info("STEP 2: TrOCR PRE-LABELING COMPLETED")
    logger.info(f"Updated {processed_count} crops in {csv_path}")
    logger.info(f"Total inference time: {total_time:.1f}s ({total_time / 60:.2f} mins)")
    if processed_count > 0:
        logger.info(f"Speed: {total_time / processed_count:.2f}s per line crop")
    logger.info("=" * 65)


def main():
    args = parse_args()
    run_prelabeling(args)


if __name__ == "__main__":
    main()
