#!/usr/bin/env python3
"""
reorder_documents.py - Safely re-numbers and reorders documents in ascending order.

Handles gaps caused by removing documents (e.g., removing non-handwritten items).
Uses a two-phase renaming strategy to avoid in-place filename collisions,
and generates an audit mapping file (reorder_mapping.json / reorder_mapping.csv)
to trace back any renumbered file to its original number.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safely reorder and renumber sequentially named documents (e.g., 1.png, 2.png, 4.png -> 1.png, 2.png, 3.png)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./redacted_documents",
        help="Target folder containing the files to reorder (default: ./redacted_documents)"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".png",
        help="File extension to match (default: .png)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index for sequence (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the reordering plan without making any changes to files"
    )
    return parser.parse_args()


def get_numeric_files(target_dir: Path, ext: str) -> List[Tuple[int, Path]]:
    """
    Finds and numerically sorts all files in target_dir matching <integer><ext>.
    Ignores non-numeric files (like manifest, gitkeep, or temporary files).
    """
    ext_clean = ext if ext.startswith(".") else f".{ext}"
    numeric_files = []

    for item in target_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ext_clean.lower():
            if item.stem.isdigit():
                numeric_files.append((int(item.stem), item))

    # Sort numerically in ascending order
    numeric_files.sort(key=lambda x: x[0])
    return numeric_files


def reorder_files(target_dir: Path, ext: str = ".png", start_idx: int = 1, dry_run: bool = False):
    target_path = Path(target_dir).resolve()
    if not target_path.exists():
        print(f"Error: Target directory '{target_path}' does not exist.")
        sys.exit(1)

    files = get_numeric_files(target_path, ext)
    total_files = len(files)

    if total_files == 0:
        print(f"No numeric {ext} files found in '{target_path}'. Nothing to do.")
        return

    ext_clean = ext if ext.startswith(".") else f".{ext}"
    print("=" * 65)
    print(f" Document Reordering Utility")
    print(f" Directory:   {target_path}")
    print(f" Total files: {total_files}")
    print(f" Current range: {files[0][0]}{ext_clean} ... {files[-1][0]}{ext_clean}")
    print(f" Target range:  {start_idx}{ext_clean} ... {start_idx + total_files - 1}{ext_clean}")
    print(f" Mode:          {'DRY RUN (preview only)' if dry_run else 'EXECUTE (renaming files)'}")
    print("=" * 65)

    # Compute planned changes
    plan: List[Dict[str, any]] = []
    unchanged_count = 0
    renamed_count = 0

    for i, (old_num, file_path) in enumerate(files):
        new_num = start_idx + i
        old_name = file_path.name
        new_name = f"{new_num}{ext_clean}"
        needs_change = (old_num != new_num)

        if needs_change:
            renamed_count += 1
        else:
            unchanged_count += 1

        plan.append({
            "path": file_path,
            "old_num": old_num,
            "new_num": new_num,
            "old_name": old_name,
            "new_name": new_name,
            "changed": needs_change
        })

    print(f"\nAnalysis Summary:")
    print(f" - Unchanged: {unchanged_count} files (already at correct sequential index)")
    print(f" - To Rename: {renamed_count} files (fill gaps in sequence)")

    if renamed_count == 0:
        print("\nAll files are already perfectly sequential! No renaming needed.")
        return

    # Show first 10 and last 5 changes as sample preview
    changed_items = [item for item in plan if item["changed"]]
    print("\nSample of planned renames:")
    for item in changed_items[:10]:
        print(f"   {item['old_name']:<15} ->  {item['new_name']}")
    if len(changed_items) > 15:
        print("   ...")
        for item in changed_items[-5:]:
            print(f"   {item['old_name']:<15} ->  {item['new_name']}")

    if dry_run:
        print("\n[DRY RUN] No files were altered. Run without --dry-run to apply changes.")
        return

    # --- Phase 1: Rename files that change to temporary intermediate names ---
    # This prevents any potential collisions (e.g. renaming 5.png -> 4.png when 4.png exists)
    print("\nPhase 1/2: Staging files with temporary names to prevent collisions...")
    temp_stage = []
    for item in plan:
        if item["changed"]:
            original_file = item["path"]
            temp_name = f"_reorder_tmp_{item['new_num']}_{item['old_num']}{ext_clean}"
            temp_path = original_file.parent / temp_name
            original_file.rename(temp_path)
            temp_stage.append((temp_path, item["new_name"]))

    # --- Phase 2: Rename temporary files to final sequential names ---
    print("Phase 2/2: Applying final sequential filenames...")
    for temp_path, final_name in temp_stage:
        final_path = temp_path.parent / final_name
        temp_path.rename(final_path)

    # --- Phase 3: Save audit mapping file ---
    mapping_json = target_path / "reorder_mapping.json"
    mapping_csv = target_path / "reorder_mapping.csv"

    # Save JSON mapping {old_name: new_name}
    mapping_dict = {item["old_name"]: item["new_name"] for item in plan}
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=2)

    # Save CSV mapping [old_name, new_name, old_num, new_num]
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_filename", "new_filename", "original_index", "new_index"])
        for item in plan:
            writer.writerow([item["old_name"], item["new_name"], item["old_num"], item["new_num"]])

    print("\n" + "=" * 65)
    print(" SUCCESS: All files successfully reordered!")
    print(f" Range: {start_idx}{ext_clean} to {start_idx + total_files - 1}{ext_clean} ({total_files} files)")
    print(f" Audit mappings saved to:")
    print(f"   - {mapping_json}")
    print(f"   - {mapping_csv}")
    print("=" * 65)


def main():
    args = parse_args()
    reorder_files(
        target_dir=Path(args.dir),
        ext=args.ext,
        start_idx=args.start,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

