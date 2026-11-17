import json
import csv
import re
from pathlib import Path

# Target directory
DATASET_DIR = Path("trocr_dataset")
MANIFEST_PATH = DATASET_DIR / "extraction_manifest.json"
METADATA_PATH = DATASET_DIR / "metadata.csv"
ANNOTATION_PATH = DATASET_DIR / "annotation_sheet.csv"
CROPS_DIR = DATASET_DIR / "crops"
BOXED_DIR = DATASET_DIR / "annotated_pages"
UPRIGHT_DIR = DATASET_DIR / "upright_pages"

# Rollback boundary: document IDs >= CUTOFF are removed
CUTOFF = 460

def get_doc_num(filename):
    m = re.match(r"^(\d+)", filename)
    return int(m.group(1)) if m else None

print(f"=== Running Boundary Cleanup for Doc IDs >= {CUTOFF} ===")

# 1. Clean Manifest
if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    initial_len = len(manifest)
    cleaned_manifest = {
        k: v for k, v in manifest.items()
        if (get_doc_num(k) or 0) < CUTOFF
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_manifest, f, indent=2)
    print(f"Manifest: {initial_len} -> {len(cleaned_manifest)} (removed {initial_len - len(cleaned_manifest)} entries)")

# 2. Clean metadata.csv
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    kept_rows = []
    removed_count = 0
    for r in rows:
        if len(r) >= 3:
            doc_name = r[2]
            dnum = get_doc_num(doc_name)
            if dnum is not None and dnum >= CUTOFF:
                removed_count += 1
                continue
        kept_rows.append(r)

    with open(METADATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(kept_rows)
    print(f"Metadata CSV: {len(rows)} -> {len(kept_rows)} (removed {removed_count} lines)")

# 3. Clean annotation_sheet.csv
if ANNOTATION_PATH.exists():
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    kept_rows = []
    removed_count = 0
    for r in rows:
        if len(r) >= 2:
            doc_name = r[1]
            dnum = get_doc_num(doc_name)
            if dnum is not None and dnum >= CUTOFF:
                removed_count += 1
                continue
        kept_rows.append(r)

    with open(ANNOTATION_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(kept_rows)
    print(f"Annotation Sheet: {len(rows)} -> {len(kept_rows)} (removed {removed_count} lines)")

# 4. Remove crops >= CUTOFF
removed_crops = 0
if CROPS_DIR.exists():
    for p in CROPS_DIR.glob("*.png"):
        dnum = get_doc_num(p.name)
        if dnum is not None and dnum >= CUTOFF:
            p.unlink(missing_ok=True)
            removed_crops += 1
print(f"Removed {removed_crops} crop files")

# 5. Remove annotated boxed pages >= CUTOFF
removed_boxed = 0
if BOXED_DIR.exists():
    for p in BOXED_DIR.glob("*.png"):
        dnum = get_doc_num(p.name)
        if dnum is not None and dnum >= CUTOFF:
            p.unlink(missing_ok=True)
            removed_boxed += 1
print(f"Removed {removed_boxed} annotated boxed page files")

# 6. Remove upright pages >= CUTOFF
removed_upright = 0
if UPRIGHT_DIR.exists():
    for p in UPRIGHT_DIR.glob("*.png"):
        dnum = get_doc_num(p.name)
        if dnum is not None and dnum >= CUTOFF:
            p.unlink(missing_ok=True)
            removed_upright += 1
print(f"Removed {removed_upright} upright page files")

print(f"=== Boundary Cleanup Complete! Safe to resume from document {CUTOFF}.png ===")
