#!/usr/bin/env python3
"""
streamlit_review.py - Step 3: Fast Local Verification & PII Removal App.

Interactive Streamlit interface for annotators to review TrOCR line crops:
  1. Displays magnified line crop with pre-filled draft OCR text.
  2. Keyboard-friendly Approve / Edit / Drop controls.
  3. Drops patient PII (names, MRNOs, IDs) and illegible scribbles from training set.
  4. Collapsible parent-page context viewer highlighting the numbered box.
  5. Auto-saves verified pairs directly to Hugging Face formatted metadata.csv.
"""

import os
import csv
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="SehatKosh TrOCR Line Verification",
    page_icon="✍️",
    layout="wide"
)

# Directories
DATASET_DIR = Path("./trocr_dataset")
CROPS_DIR = DATASET_DIR / "crops"
ANNOTATED_PAGES_DIR = DATASET_DIR / "annotated_pages"
ANNOTATION_SHEET = DATASET_DIR / "annotation_sheet.csv"
METADATA_CSV = DATASET_DIR / "metadata.csv"


@st.cache_data(ttl=2)
def load_data():
    if not ANNOTATION_SHEET.exists():
        return pd.DataFrame(), pd.DataFrame()

    df_ann = pd.read_csv(ANNOTATION_SHEET, dtype=str).fillna("")
    if "status" not in df_ann.columns:
        # Determine initial status
        df_ann["status"] = df_ann["verified_text"].apply(
            lambda x: "Verified" if x.strip() and not x.startswith("[DISCARD") 
            else ("Discarded" if x.startswith("[DISCARD") else "Pending")
        )

    df_meta = pd.DataFrame()
    if METADATA_CSV.exists():
        df_meta = pd.read_csv(METADATA_CSV, dtype=str).fillna("")

    return df_ann, df_meta


def save_record(idx: int, verified_text: str, status: str):
    df_ann, df_meta = load_data()
    if df_ann.empty or idx >= len(df_ann):
        return

    # Update annotation sheet
    df_ann.at[idx, "verified_text"] = verified_text
    df_ann.at[idx, "status"] = status
    df_ann.to_csv(ANNOTATION_SHEET, index=False)

    # If verified (and not discarded), update metadata.csv
    crop_file = df_ann.at[idx, "crop_file"]
    rel_path = f"crops/{crop_file}"

    if not df_meta.empty:
        mask = df_meta["file_name"] == rel_path
        if mask.any():
            if status == "Verified":
                df_meta.loc[mask, "text"] = verified_text
            else:
                # Remove discarded line from training metadata
                df_meta = df_meta[~mask]
            df_meta.to_csv(METADATA_CSV, index=False)


# Main App Logic
df_ann, df_meta = load_data()

if df_ann.empty:
    st.warning(f"No annotation sheet found at `{ANNOTATION_SHEET.resolve()}`. Run Step 1 (`extract_trocr_lines.py`) first to generate dataset crops.")
    st.stop()

# Sidebar: Navigator & Metrics
st.sidebar.title("✍️ TrOCR Dataset Review")

total_crops = len(df_ann)
verified_count = (df_ann["status"] == "Verified").sum()
discarded_count = (df_ann["status"] == "Discarded").sum()
pending_count = total_crops - verified_count - discarded_count

col_m1, col_m2, col_m3 = st.sidebar.columns(3)
col_m1.metric("Verified", verified_count)
col_m2.metric("Discarded", discarded_count)
col_m3.metric("Pending", pending_count)

progress_pct = int((verified_count + discarded_count) / max(1, total_crops) * 100)
st.sidebar.progress(progress_pct / 100, text=f"Overall Progress: {progress_pct}%")

# Filter Mode
filter_mode = st.sidebar.radio("View Filter", ["Only Pending", "All Crops", "Only Verified", "Only Discarded"])

if filter_mode == "Only Pending":
    display_indices = df_ann[df_ann["status"] == "Pending"].index.tolist()
elif filter_mode == "Only Verified":
    display_indices = df_ann[df_ann["status"] == "Verified"].index.tolist()
elif filter_mode == "Only Discarded":
    display_indices = df_ann[df_ann["status"] == "Discarded"].index.tolist()
else:
    display_indices = df_ann.index.tolist()

if not display_indices:
    st.info(f"No line crops match filter: **{filter_mode}**.")
    st.stop()

# Session state navigation
if "curr_pos" not in st.session_state:
    st.session_state.curr_pos = 0

if st.session_state.curr_pos >= len(display_indices):
    st.session_state.curr_pos = max(0, len(display_indices) - 1)

# Jump selector
selected_pos = st.sidebar.number_input(
    "Jump to Item (within filter)",
    min_value=1,
    max_value=len(display_indices),
    value=st.session_state.curr_pos + 1,
    step=1
) - 1

if selected_pos != st.session_state.curr_pos:
    st.session_state.curr_pos = selected_pos

active_idx = display_indices[st.session_state.curr_pos]
curr_row = df_ann.iloc[active_idx]

crop_filename = curr_row["crop_file"]
source_doc = curr_row["source_doc"]
line_num = curr_row["line_num"]
draft_text = curr_row["draft_ocr"]
existing_verified = curr_row["verified_text"]
item_status = curr_row.get("status", "Pending")

# Status Badge
if item_status == "Verified":
    badge = "🟢 **VERIFIED (IN TRAINING SET)**"
elif item_status == "Discarded":
    badge = "🔴 **DISCARDED (PII / UNREADABLE)**"
else:
    badge = "🟡 **PENDING REVIEW**"

st.markdown(f"### Crop: `{crop_filename}` (Item {st.session_state.curr_pos + 1} of {len(display_indices)}) | {badge}")
st.caption(f"Source Document: `{source_doc}` | Line Number: `#{line_num}`")

crop_path = CROPS_DIR / crop_filename

# Layout: Review Workspace
col_img, col_form = st.columns([1.2, 1.8])

with col_img:
    st.subheader("🔍 Cropped Handwritten Line")
    if crop_path.exists():
        img = Image.open(crop_path)
        # Display magnified crop
        st.image(img, use_container_width=True)
    else:
        st.error(f"Image crop not found at {crop_path}")

with col_form:
    st.subheader("✏️ Ground-Truth Text Editor")
    default_val = existing_verified if existing_verified and not existing_verified.startswith("[DISCARD") else draft_text
    
    with st.form(key=f"form_{active_idx}"):
        user_text = st.text_input(
            "Verify or edit medical text (Press Enter to approve):",
            value=default_val,
            key=f"input_{active_idx}",
            help="Fix any spelling errors. Discard if this contains patient name, CNIC, phone number, or is completely illegible."
        )

        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 1])
        
        with btn_col1:
            approve_submitted = st.form_submit_button("✅ Approve & Next", type="primary", use_container_width=True)
        with btn_col2:
            discard_submitted = st.form_submit_button("🗑️ Discard (PII / Illegible)", use_container_width=True)
        with btn_col3:
            skip_submitted = st.form_submit_button("⏭️ Skip", use_container_width=True)

    if approve_submitted:
        clean_text = user_text.strip()
        if clean_text:
            save_record(active_idx, clean_text, "Verified")
            if st.session_state.curr_pos < len(display_indices) - 1:
                st.session_state.curr_pos += 1
            st.rerun()
        else:
            st.warning("Cannot approve empty text. If line should be excluded, click Discard.")

    if discard_submitted:
        save_record(active_idx, "[DISCARDED_PII_OR_UNREADABLE]", "Discarded")
        if st.session_state.curr_pos < len(display_indices) - 1:
            st.session_state.curr_pos += 1
        st.rerun()

    if skip_submitted:
        if st.session_state.curr_pos < len(display_indices) - 1:
            st.session_state.curr_pos += 1
        st.rerun()

# Navigation Row
nav_col1, nav_col2, _ = st.columns([1.5, 1.5, 4])
with nav_col1:
    if st.button("⬅ Previous Item", use_container_width=True):
        if st.session_state.curr_pos > 0:
            st.session_state.curr_pos -= 1
            st.rerun()
with nav_col2:
    if st.button("Next Item ➡", use_container_width=True):
        if st.session_state.curr_pos < len(display_indices) - 1:
            st.session_state.curr_pos += 1
            st.rerun()

# Collapsible Full-Page Context
st.markdown("---")
with st.expander(f"📄 View Full Parent Document Context (`{source_doc}` - Line #{line_num})", expanded=False):
    source_stem = Path(source_doc).stem
    annotated_page_path = ANNOTATED_PAGES_DIR / f"{source_stem}_boxed.png"
    
    if annotated_page_path.exists():
        st.info(f"Looking for red box **#{line_num}** on page `{source_doc}`:")
        page_img = Image.open(annotated_page_path)
        st.image(page_img, use_container_width=True)
    else:
        st.warning(f"Verification overlay image `{annotated_page_path.name}` not found.")

# Sidebar Export
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Dataset")
if METADATA_CSV.exists():
    with open(METADATA_CSV, "rb") as f:
        st.sidebar.download_button(
            "Download metadata.csv (HuggingFace)",
            data=f.read(),
            file_name="metadata.csv",
            mime="text/csv",
            use_container_width=True
        )
