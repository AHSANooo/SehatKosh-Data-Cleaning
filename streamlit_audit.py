#!/usr/bin/env python3
"""
streamlit_audit.py - Rapid Human-in-the-Loop Audit Application
for Medical Prescriptions and Receipts De-identification.

Run locally:
    streamlit run streamlit_audit.py
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
import streamlit as st
from PIL import Image

def natural_key(p: Path):
    """Natural sort key to sort 1.png, 2.png, ..., 10.png correctly."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.name)]

# Page setup
st.set_page_config(
    page_title="SehatKosh De-identification Audit",
    page_icon="🛡️",
    layout="wide"
)

# Directories
RAW_DIR = Path("./Data")
OUTPUT_DIR = Path("./redacted_documents")
MANIFEST_FILE = OUTPUT_DIR / "redaction_manifest.json"
AUDIT_LOG_FILE = OUTPUT_DIR / "audit_review.csv"

# Load Manifest
@st.cache_data(ttl=5)
def load_manifest():
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# Load Audit Decisions
def load_audit_log() -> pd.DataFrame:
    if AUDIT_LOG_FILE.exists():
        try:
            return pd.read_csv(AUDIT_LOG_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["filename", "status", "notes", "timestamp"])

def save_audit_record(filename: str, status: str, notes: str):
    df = load_audit_log()
    # Replace existing or append
    if filename in df["filename"].values:
        df.loc[df["filename"] == filename, ["status", "notes", "timestamp"]] = [
            status, notes, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    else:
        new_row = pd.DataFrame([{
            "filename": filename,
            "status": status,
            "notes": notes,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(AUDIT_LOG_FILE, index=False)


# Find all output redacted images
redacted_files = sorted([f for f in OUTPUT_DIR.glob("*.png") if not f.name.endswith("_original.png")], key=natural_key)

if not redacted_files:
    st.warning(f"No redacted documents found in `{OUTPUT_DIR.resolve()}`. Run `batch_redact.py` first to generate redacted images.")
    st.stop()

# Initialize session state
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "notes_input" not in st.session_state:
    st.session_state.notes_input = ""

manifest = load_manifest()
audit_df = load_audit_log()
audited_dict = dict(zip(audit_df["filename"], audit_df["status"])) if not audit_df.empty else {}

# Sidebar
st.sidebar.title("🛡️ Audit Navigator")
st.sidebar.markdown(f"**Total Redacted Documents:** {len(redacted_files)}")

# Audit Progress Metrics
approved_count = sum(1 for s in audited_dict.values() if s == "Approved")
flagged_count = sum(1 for s in audited_dict.values() if s == "Flagged")
pending_count = len(redacted_files) - approved_count - flagged_count

col_m1, col_m2, col_m3 = st.sidebar.columns(3)
col_m1.metric("Approved", approved_count)
col_m2.metric("Flagged", flagged_count)
col_m3.metric("Pending", pending_count)

# Filter option
filter_mode = st.sidebar.radio("View Filter", ["All", "Only Flagged", "Only Pending"])
if filter_mode == "Only Flagged":
    display_files = [f for f in redacted_files if audited_dict.get(f.name) == "Flagged"]
elif filter_mode == "Only Pending":
    display_files = [f for f in redacted_files if f.name not in audited_dict]
else:
    display_files = redacted_files

if not display_files:
    st.info(f"No documents match filter: {filter_mode}")
    st.stop()

if st.session_state.idx >= len(display_files):
    st.session_state.idx = 0

st.session_state.idx = st.sidebar.number_input(
    "Jump to Document Index",
    min_value=0,
    max_value=len(display_files) - 1,
    value=st.session_state.idx,
    step=1
)

curr_file = display_files[st.session_state.idx]
current_status = audited_dict.get(curr_file.name, "Pending")

# Status Badge
if current_status == "Approved":
    badge = "🟢 **APPROVED**"
elif current_status == "Flagged":
    badge = "🔴 **FLAGGED FOR REVIEW**"
else:
    badge = "🟡 **PENDING AUDIT**"

st.markdown(f"### Document: `{curr_file.name}` ({st.session_state.idx + 1} of {len(display_files)}) | Status: {badge}")

# Metadata summary from manifest
manifest_info = manifest.get(curr_file.name.replace(".png", ".pdf"), {})
if not manifest_info:
    manifest_info = manifest.get(curr_file.stem + ".pdf", {})

if manifest_info:
    st.caption(
        f"⏱️ Processing time: {manifest_info.get('duration_sec', 'N/A')}s | "
        f"🔲 PII Boxes Redacted: {manifest_info.get('boxes_redacted', 'N/A')} | "
        f"📅 Date: {manifest_info.get('timestamp', 'N/A')}"
    )

# Side-by-Side Comparison
col_left, col_right = st.columns([1, 1])

# Check for original file
orig_file = None
stem_clean = curr_file.stem.split("_page")[0]
for ext in [".pdf", ".jpg", ".jpeg", ".png"]:
    candidate = RAW_DIR / f"{stem_clean}{ext}"
    if candidate.exists():
        orig_file = candidate
        break

with col_left:
    st.subheader("📄 Original Document")
    if orig_file:
        if orig_file.suffix.lower() == ".pdf":
            # Quick render original PDF page
            try:
                import fitz
                doc = fitz.open(str(orig_file))
                page_idx = 0
                if "_page" in curr_file.stem:
                    try:
                        page_idx = int(curr_file.stem.split("_page")[1]) - 1
                    except Exception:
                        page_idx = 0
                page_idx = min(page_idx, len(doc) - 1)
                pix = doc[page_idx].get_pixmap(dpi=150)
                st.image(pix.tobytes(), use_column_width=True)
                doc.close()
            except Exception:
                try:
                    from pdf2image import convert_from_path
                    pages = convert_from_path(str(orig_file), dpi=150, first_page=1, last_page=1)
                    if pages:
                        st.image(pages[0], use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not preview original PDF: {e}")
        else:
            st.image(Image.open(orig_file), use_column_width=True)
    else:
        st.info("Original raw document not found in ./Data")

with col_right:
    st.subheader("🛡️ De-Identified Output")
    redacted_image = Image.open(curr_file)
    st.image(redacted_image, use_column_width=True)

# Controls & Feedback
st.markdown("---")
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 2, 3])

with ctrl_col1:
    if st.button("⬅ Previous Document", use_container_width=True):
        if st.session_state.idx > 0:
            st.session_state.idx -= 1
            st.rerun()

with ctrl_col2:
    if st.button("Next Document ➡", use_container_width=True):
        if st.session_state.idx < len(display_files) - 1:
            st.session_state.idx += 1
            st.rerun()

with ctrl_col3:
    if st.button("✅ Approve & Next", type="primary", use_container_width=True):
        save_audit_record(curr_file.name, "Approved", st.session_state.notes_input)
        st.session_state.notes_input = ""
        if st.session_state.idx < len(display_files) - 1:
            st.session_state.idx += 1
        st.rerun()

with ctrl_col4:
    flag_colA, flag_colB = st.columns([1, 1])
    with flag_colA:
        if st.button("⚠️ Flag & Next", use_container_width=True):
            save_audit_record(curr_file.name, "Flagged", st.session_state.notes_input)
            st.session_state.notes_input = ""
            if st.session_state.idx < len(display_files) - 1:
                st.session_state.idx += 1
            st.rerun()
    with flag_colB:
        notes = st.text_input("Audit Notes (optional)", key="notes_input", placeholder="e.g. Missed name in header")

# Sidebar Export
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Audit Report")
if not audit_df.empty:
    csv_bytes = audit_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download audit_review.csv",
        data=csv_bytes,
        file_name="audit_review.csv",
        mime="text/csv",
        use_container_width=True
    )

