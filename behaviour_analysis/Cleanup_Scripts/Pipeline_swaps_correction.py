"""
Full pipeline script to process ONE video end-to-end:
  Step 1: assign_subjects_pipeline.py (SAM3 mask-based subject assignment)
  Step 2: anatomical_correction.py    (anatomical correction: ID swaps only, no interpolation)
  Step 3: mask_cleanup.py             (remove keypoints outside their own SAM3 mask, no interpolation)
  Step 4: visualize_corrected.py      (verification video, built from the FINAL cleaned data)

HOW TO USE:
    1. Edit the paths in the CONFIG section below with Notepad
       (or any text editor) before running.
    2. Save the file.
    3. Run from Anaconda prompt:
           python path\to\Pipeline_swaps_correction.py

OUTPUT (inside OUTPUT_DIR):
    - *_assignment_raw.csv          (intermediate: raw assignment, input for step 2)
    - *_corrected_anatomical.csv    (result of step 1+2 — anatomical swap correction)
    - *_cleaned.csv                 (FINAL data — mask-based cleanup, in wide DLC-style
                                      format, ready to use as a deepOF tracking input)
    - *_mask_cleanup_report.xlsx    (per-joint removed/kept stats + detail list)
    - *_corrected.mp4               (verification video, from the final cleaned data)
"""

import subprocess
import sys
from pathlib import Path

# =====================================================================
# CONFIG — EDIT THESE PATHS BEFORE RUNNING (use Notepad to change them)
# =====================================================================

PRED_CSV = r"/path/to/dlc/csv"         # DLC predicted keypoints CSV
SHARDS_DIR = r"/path/to/shards/folder" # folder containing the shards
CENTROIDS  = r"/path/to/shards/folder/centroids.npz"        # SAM3 centroids file
VIDEO_IN   = r"/path/to/video.mp4"  # original video
OUTPUT_DIR = r"/path/to/output/folder"       # where results will be saved

# Segments to include in the verification video.
# Leave as None to process the WHOLE video.
# Example to use only specific parts: "0-300,7100-8900"
SEGMENTS = None

# Subject names and colors used in the verification video:
#   SUBJ1 = "stim"  -> drawn in CYAN
#   SUBJ2 = "exp"   -> drawn in YELLOW
SUBJ1 = "stim"   # cyan
SUBJ2 = "exp"    # yellow

# Tolerance (in pixels) used by the mask cleanup step (Step 3)
MASK_TOLERANCE_PX = 15

# Value used for the "scorer" header level in the final DLC-style wide CSV (Step 3 output)
SCORER_NAME = "corrected_tracking"

# =====================================================================
# You should not need to edit anything below this line
# =====================================================================


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f">> {desc}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(PRED_CSV).stem

    raw_csv     = out_dir / f"{stem}_assignment_raw.csv"
    anat_csv    = out_dir / f"{stem}_corrected_anatomical.csv"
    cleaned_csv = out_dir / f"{stem}_cleaned.csv"
    video_out   = out_dir / f"{stem}_corrected.mp4"

    # Locate the other scripts in the same folder as this file
    script_dir = Path(__file__).parent
    step1 = script_dir / "assign_subjects_pipeline.py"
    step2 = script_dir / "anatomical_correction.py"
    step3 = script_dir / "mask_cleanup.py"
    step4 = script_dir / "visualize_corrected.py"

    # Step 1: subject assignment based on SAM3 masks
    run([sys.executable, str(step1),
         "--pred_csv",   PRED_CSV,
         "--shards_dir", SHARDS_DIR,
         "--centroids",  CENTROIDS,
         "--output_dir", str(out_dir)],
        "Step 1: subject assignment based on SAM3 masks")

    # Step 2: anatomical correction (swap only, no interpolation)
    run([sys.executable, str(step2),
         "--input_csv",  str(raw_csv),
         "--output_csv", str(anat_csv)],
        "Step 2: anatomical correction (swap only)")

    # Step 3: mask-based cleanup (removes points outside their own SAM3 mask, no interpolation, reports)
    run([sys.executable, str(step3),
         "--input_csv",    str(anat_csv),
         "--shards_dir",   SHARDS_DIR,
         "--output_dir",   str(out_dir),
         "--tolerance_px", str(MASK_TOLERANCE_PX),
         "--scorer_name",  SCORER_NAME],
        "Step 3: mask-based cleanup")

    # Step 4: verification video — built from the FINAL cleaned data only
    cmd4 = [sys.executable, str(step4),
            "--assignment_csv", str(cleaned_csv),
            "--video_in",       VIDEO_IN,
            "--video_out",      str(video_out),
            "--subj1",          SUBJ1,
            "--subj2",          SUBJ2,
            "--skeleton"]
    if SEGMENTS:
        cmd4 += ["--segments", SEGMENTS]
    run(cmd4, "Step 4: generating verification video (from cleaned data)")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"  Step 1+2 result:      {anat_csv}")
    print(f"  Final (deepOF-ready): {cleaned_csv}")
    print(f"  Cleanup report:       {out_dir / (stem + '_mask_cleanup_report.xlsx')}")
    print(f"  Verification video:   {video_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
