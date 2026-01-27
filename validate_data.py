#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
from collections import defaultdict

def read_ppm_binary(file_path):
    with open(file_path, 'rb') as f:
        return f.read()

def validate_directory(testruns_dir):
    testruns_path = Path(testruns_dir)
    testrun_dirs = sorted([d for d in testruns_path.iterdir() if d.is_dir() and d.name.startswith('testrun_')])

    print(f"Found {len(testrun_dirs)} testrun directories")

    model_frame_hits = defaultdict(lambda: defaultdict(set))
    # Only store the first PPM for each model/cam_key for comparison
    model_frame_first_ppm = defaultdict(lambda: defaultdict(lambda: None))

    for testrun_idx, testrun_dir in enumerate(testrun_dirs, 1):
        print(f"\nProcessing testrun {testrun_idx}/{len(testrun_dirs)}: {testrun_dir.name}")

        shading_csv = testrun_dir / 'shading_times.csv'
        if not shading_csv.exists():
            print(f"  Skipping: no shading_times.csv found")
            continue

        with open(shading_csv, 'r') as f:
            reader = csv.DictReader(f)
            frame_idx = 0
            for row in reader:
                model = row['model_name']
                cam_key = (row['cam_pos_x'], row['cam_pos_y'], row['cam_pos_z'])
                hits = row['time_seconds']

                model_frame_hits[model][cam_key].add(hits)

                ppm_file = testrun_dir / f'screen_{frame_idx}.ppm'
                if ppm_file.exists():
                    print(f"  Frame {frame_idx}: model={model}, cam={cam_key[:3]}...")

                    # Read current PPM
                    ppm_data = read_ppm_binary(ppm_file)

                    # Compare with first PPM if it exists
                    if model_frame_first_ppm[model][cam_key] is None:
                        # Store first PPM for this model/cam_key
                        model_frame_first_ppm[model][cam_key] = ppm_data
                        print(f"    Stored as reference image")
                    else:
                        # Compare with stored first PPM
                        if ppm_data != model_frame_first_ppm[model][cam_key]:
                            print(f"FAILED: Model {model} at frame {cam_key} has different PPM images")
                            return False
                        print(f"    Image matches reference")

                    # Clear current ppm_data to free memory
                    del ppm_data

                frame_idx += 1

    print(f"\nValidating hit counts across all frames...")
    for model in model_frame_hits:
        for cam_key in model_frame_hits[model]:
            hits_set = model_frame_hits[model][cam_key]
            if len(hits_set) != 1:
                print(f"FAILED: Model {model} at frame {cam_key} has inconsistent hit counts: {hits_set}")
                return False

    print(f"All validations passed!")
    return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <testruns_directory>")
        sys.exit(1)

    if validate_directory(sys.argv[1]):
        print("successful")
    else:
        sys.exit(1)
