#!/usr/bin/env python3
"""
train_trocr.py - Step 4: Fine-Tuning TrOCR on Verified Medical Handwriting.

Loads verified (crop, text) pairs from trocr_dataset/metadata.csv and fine-tunes
microsoft/trocr-base-handwritten using Hugging Face Transformers & Seq2SeqTrainer.
Optimized for 4GB NVIDIA GPUs with FP16, gradient checkpointing, and accumulation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator,
    )
except ImportError:
    torch = None
    Dataset = object

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_trocr")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 4: Fine-Tune TrOCR on Verified Medical Handwriting"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./trocr_dataset",
        help="Path to trocr_dataset folder containing metadata.csv and crops/ (default: ./trocr_dataset)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="HuggingFace model identifier (default: microsoft/trocr-base-handwritten)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trocr_medical_finetuned",
        help="Directory to save fine-tuned model checkpoints (default: ./trocr_medical_finetuned)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2 for 4GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--test_split", type=float, default=0.1, help="Validation split fraction (default: 0.1)")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum target sequence length (default: 64)")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps (default: 200)")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps (default: 200)")
    return parser.parse_args()


class MedicalHandwritingDataset(Dataset):
    """PyTorch Dataset loading image crops and tokenized target texts."""

    def __init__(self, root_dir: Path, df: pd.DataFrame, processor: TrOCRProcessor, max_length: int = 64):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_rel_path = row["file_name"]
        text = str(row["text"]).strip()

        img_path = self.root_dir / image_rel_path
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize labels
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Important: replace padding token id with -100 so loss is not computed on pad tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def compute_cer(preds, labels):
    """Character Error Rate computation."""
    import editdistance
    total_dist = 0
    total_len = 0
    for p, l in zip(preds, labels):
        total_dist += editdistance.eval(p, l)
        total_len += max(1, len(l))
    return total_dist / float(total_len)


def main():
    args = parse_args()

    if torch is None:
        logger.error("PyTorch and Hugging Face Transformers are required for Step 4.")
        logger.error("To install, run: pip install torch torchvision transformers editdistance")
        sys.exit(1)
    dataset_dir = Path(args.dataset_dir).resolve()
    metadata_path = dataset_dir / "metadata.csv"

    if not metadata_path.exists():
        logger.error(f"Cannot find metadata file at {metadata_path}. Complete Step 3 verification first.")
        sys.exit(1)

    # 1. Load verified data
    df = pd.read_csv(metadata_path, dtype=str).fillna("")
    # Filter only verified rows with text and valid crops
    df_clean = df[
        (df["text"].str.strip() != "") & 
        (~df["text"].str.startswith("[DISCARD"))
    ].copy()

    logger.info(f"Loaded {len(df_clean)} verified handwriting pairs from {metadata_path}")
    if len(df_clean) < 10:
        logger.error("Too few verified records to begin training. Please verify more crops in Step 3.")
        sys.exit(1)

    # Verify that image crops actually exist on disk
    valid_mask = df_clean["file_name"].apply(lambda p: (dataset_dir / p).exists())
    df_clean = df_clean[valid_mask]
    logger.info(f"Verified {len(df_clean)} image crops present on disk.")

    # 2. Train / Validation Split
    shuffled = df_clean.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = max(5, int(len(shuffled) * args.test_split))
    val_df = shuffled.iloc[:val_size]
    train_df = shuffled.iloc[val_size:]

    logger.info(f"Dataset split: {len(train_df)} training samples | {len(val_df)} validation samples")

    # 3. Load TrOCR Processor & Model
    logger.info(f"Loading pretrained TrOCR model: {args.model_name}...")
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    # Configure special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Generation hyperparameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Memory optimization: gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for memory optimization.")

    train_dataset = MedicalHandwritingDataset(dataset_dir, train_df, processor, max_length=args.max_length)
    val_dataset = MedicalHandwritingDataset(dataset_dir, val_df, processor, max_length=args.max_length)

    # Metric evaluation function
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        try:
            cer = compute_cer(pred_str, label_str)
        except Exception:
            cer = 1.0

        return {"cer": cer}

    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fp16=use_fp16,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=25,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0 if sys.platform.startswith("win") else 2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    logger.info("Starting TrOCR fine-tuning...")
    trainer.train()

    # Save final best model and processor
    out_path = Path(args.output_dir)
    trainer.save_model(str(out_path / "best_model"))
    processor.save_pretrained(str(out_path / "best_model"))
    logger.info(f"Model and processor saved successfully to {out_path / 'best_model'}")


if __name__ == "__main__":
    main()
