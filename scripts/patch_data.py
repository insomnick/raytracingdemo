#!/usr/bin/env python3
"""
Script to patch testrun data by copying a base directory and applying patches.

Usage:
    python3 patch_data.py <base_dir> <new_dir> <patch_dir1> [patch_dir2 ...]

Example:
    python3 patch_data.py testruns_2026_02_04 testruns_patched testruns_2026_02_04_patch testruns_2026_02_04_patch_2
"""

import sys
import os
import shutil
import csv
from pathlib import Path
from typing import Dict, Tuple, List, Set


def get_signature(row: Dict[str, str]) -> Tuple:
    """
    Extract the signature (common fields) from a CSV row.
    Signature: (file_name, model_name, model_scale, algorithm_name)
    """
    return (
        row['file_name'],
        row['model_name'],
        row['model_scale'],
        row['algorithm_name']
    )


def get_camera_tuple(row: Dict[str, str]) -> Tuple:
    """
    Extract the camera position/direction tuple from a CSV row.
    Camera: (cam_pos_x, cam_pos_y, cam_pos_z, cam_dir_x, cam_dir_y, cam_dir_z)
    """
    return (
        row['cam_pos_x'],
        row['cam_pos_y'],
        row['cam_pos_z'],
        row['cam_dir_x'],
        row['cam_dir_y'],
        row['cam_dir_z']
    )


def load_csv_data(csv_path: Path) -> List[Dict[str, str]]:
    """Load CSV file and return list of row dictionaries."""
    if not csv_path.exists():
        return []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv_data(csv_path: Path, rows: List[Dict[str, str]], fieldnames: List[str]):
    """Save data to CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_testrun_fingerprint(testrun_dir: Path) -> frozenset:
    """
    Get a fingerprint for a testrun directory based on all CSV content except time_seconds.
    Returns a frozenset of (csv_filename, signature, camera_tuple) tuples.
    """
    fingerprint = set()

    csv_files = sorted(testrun_dir.glob('*.csv'))
    for csv_file in csv_files:
        rows = load_csv_data(csv_file)
        for row in rows:
            sig = get_signature(row)
            cam = get_camera_tuple(row)
            fingerprint.add((csv_file.name, sig, cam))

    return frozenset(fingerprint)


def get_testrun_data(testrun_dir: Path) -> Dict[Tuple, Dict[Tuple, str]]:
    """
    Get all data from a testrun directory.
    Returns: Dict[csv_filename] -> Dict[(signature, camera_tuple)] -> time_seconds
    """
    testrun_data = {}

    csv_files = sorted(testrun_dir.glob('*.csv'))
    for csv_file in csv_files:
        csv_name = csv_file.name
        csv_data = {}

        rows = load_csv_data(csv_file)
        for row in rows:
            sig = get_signature(row)
            cam = get_camera_tuple(row)
            csv_data[(sig, cam)] = row['time_seconds']

        if csv_data:
            testrun_data[csv_name] = csv_data

    return testrun_data


def apply_patches_to_directory(target_dir: Path, patch_dirs: List[Path]):
    """
    Apply patches to the target directory in order.
    Matches testrun directories by their content fingerprint with 1:1 mapping.
    """
    print(f"\nStep 1: Building fingerprints for target testruns...")
    target_testruns = sorted([d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith('testrun_')])
    print(f"Found {len(target_testruns)} target testruns")

    # Build fingerprint map for target - each fingerprint maps to a LIST of testruns
    target_fingerprints = {}
    for testrun_dir in target_testruns:
        fingerprint = get_testrun_fingerprint(testrun_dir)
        if fingerprint not in target_fingerprints:
            target_fingerprints[fingerprint] = []
        target_fingerprints[fingerprint].append(testrun_dir)

    print(f"Created {len(target_fingerprints)} unique fingerprints")

    # Show distribution
    max_count = max(len(v) for v in target_fingerprints.values()) if target_fingerprints else 0
    print(f"Testruns per fingerprint: {max_count} (expecting 10 for repetitions)")

    # Track which target testruns have been matched (for 1:1 mapping)
    matched_targets = set()

    # Process each patch directory in order
    total_matched = 0
    total_updated = 0

    for patch_idx, patch_dir in enumerate(patch_dirs, 1):
        print(f"\nStep 2.{patch_idx}: Processing patch directory: {patch_dir.name}")

        patch_testruns = sorted([d for d in patch_dir.iterdir() if d.is_dir() and d.name.startswith('testrun_')])
        print(f"Found {len(patch_testruns)} patch testruns")

        matched_count = 0
        updated_rows = 0

        for patch_testrun in patch_testruns:
            # Get fingerprint of patch testrun
            patch_fingerprint = get_testrun_fingerprint(patch_testrun)

            # Find matching target testrun (1:1 mapping - find first unmatched)
            if patch_fingerprint not in target_fingerprints:
                continue

            # Find first unmatched target testrun with this fingerprint
            target_testrun = None
            for candidate in target_fingerprints[patch_fingerprint]:
                if candidate not in matched_targets:
                    target_testrun = candidate
                    matched_targets.add(candidate)
                    break

            if target_testrun is None:
                # All testruns with this fingerprint already matched
                continue

            matched_count += 1

            # Get patch data
            patch_data = get_testrun_data(patch_testrun)

            # Apply patches to all CSV files in target testrun
            for csv_name, patch_csv_data in patch_data.items():
                target_csv_path = target_testrun / csv_name

                if not target_csv_path.exists():
                    continue

                # Load target CSV
                rows = load_csv_data(target_csv_path)
                if not rows:
                    continue

                fieldnames = list(rows[0].keys())

                # Update rows with patch data
                for row in rows:
                    sig = get_signature(row)
                    cam = get_camera_tuple(row)
                    key = (sig, cam)

                    if key in patch_csv_data:
                        row['time_seconds'] = patch_csv_data[key]
                        updated_rows += 1

                # Save updated CSV
                save_csv_data(target_csv_path, rows, fieldnames)

        print(f"  Matched {matched_count} testruns")
        print(f"  Updated {updated_rows} rows")

        total_matched += matched_count
        total_updated += updated_rows

    print(f"\n✓ Patching complete!")
    print(f"  Total testruns matched: {total_matched} / {len(target_testruns)}")
    print(f"  Total rows updated: {total_updated}")

    if total_matched < len(target_testruns):
        unmatched_count = len(target_testruns) - total_matched
        print(f"  WARNING: {unmatched_count} testruns were not patched (no matching patch found)")
    else:
        print(f"  ✓ All testruns were successfully patched!")


def main():
    if len(sys.argv) < 4:
        print("Error: Insufficient arguments", file=sys.stderr)
        print("\nUsage:", file=sys.stderr)
        print("  python3 patch_data.py <base_dir> <new_dir> <patch_dir1> [patch_dir2 ...]", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python3 patch_data.py testruns_2026_02_04 testruns_patched testruns_2026_02_04_patch testruns_2026_02_04_patch_2", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    new_dir = Path(sys.argv[2])
    patch_dirs = [Path(arg) for arg in sys.argv[3:]]

    # Validate inputs
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    if not base_dir.is_dir():
        print(f"Error: Base path is not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    for patch_dir in patch_dirs:
        if not patch_dir.exists():
            print(f"Error: Patch directory does not exist: {patch_dir}", file=sys.stderr)
            sys.exit(1)
        if not patch_dir.is_dir():
            print(f"Error: Patch path is not a directory: {patch_dir}", file=sys.stderr)
            sys.exit(1)

    if new_dir.exists():
        print(f"Error: Target directory already exists: {new_dir}", file=sys.stderr)
        print("Please remove it or choose a different name.", file=sys.stderr)
        sys.exit(1)

    print("="*60)
    print("Testrun Data Patcher")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"New directory:  {new_dir}")
    print(f"Patch directories ({len(patch_dirs)}):")
    for i, patch_dir in enumerate(patch_dirs, 1):
        print(f"  {i}. {patch_dir}")
    print("="*60)

    # Step 1: Copy base directory to new directory
    print(f"\nCopying base directory to new location...")
    shutil.copytree(base_dir, new_dir)
    print(f"✓ Copy complete")

    # Step 2: Apply patches in order
    apply_patches_to_directory(new_dir, patch_dirs)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
