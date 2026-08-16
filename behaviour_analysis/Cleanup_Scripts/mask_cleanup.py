"""
Step 3: SAM3 mask-based cleanup.

For every keypoint (after anatomical correction), checks whether the point falls
inside the SAM3 mask of the subject it is assigned to (with a pixel tolerance for
mask edges). If the point is outside its own mask, it is removed:
    - x, y       -> set to NaN
    - likelihood -> set to 0.0 (the original DLC likelihood is kept for every
                    point that is NOT removed, so real per-point confidence
                    values are preserved everywhere except at points this step
                    actively rejects)

Joints listed in NEVER_DELETE_JOINTS (currently: nose) are never removed by this
step, regardless of whether they fall outside their own mask -- they are marked
"kept_protected" and left untouched.

Produces:
    - <name>_cleaned.csv               : FINAL data, in the same wide, multi-row-header
                                          format as the original DLC CSV (scorer / individuals
                                          / bodyparts / coords), ready to be used as a deepOF
                                          tracking input file.
    - <name>_mask_cleanup_report.xlsx  : Excel report with statistics
        - Sheet "Summary": for each joint/subject, total checked, removed, percentage
        - Sheet "Removed Details": list of every removed point (frame, subject, joint,
          original coordinates, original likelihood)

NOTE: the *_corrected_anatomical.csv file produced by step 2 is NOT modified/overwritten.
This script reads from it and writes a separate new file.

NOTE ON DEEPOF COMPATIBILITY: this script rebuilds a standard DLC-style wide CSV so that
deepOF's "csv" table format loader can read it directly. This has not been verified against
deepOF's actual parser on your machine — test it on a small file with
deepof.data.Project(...) before relying on it for the full analysis.

USAGE:
    python mask_cleanup.py \
        --input_csv   path/to/*_corrected_anatomical.csv \
        --shards_dir  path/to/folder/with/shard_*.npz \
        --output_dir  path/to/output/folder \
        [--tolerance_px 15] \
        [--scorer_name corrected_tracking]
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_TO_OBJ = {"stim": 1, "exp": 2}

# Joints listed here are NEVER removed by the mask check, no matter how far outside
# their own mask they appear to be. Add/remove joint names here as needed.
NEVER_DELETE_JOINTS = {"nose"}

ALL_JOINTS = [
    "nose", "right_ear", "left_ear", "spine_1", "center", "spine_2",
    "right_fhip", "left_fhip", "right_bhip", "left_bhip",
    "tail_base", "tail_1", "tail_2", "tail_tip",
]


def decode_mask(shard_data, frame_id, obj):
    shape = shard_data[f"frame{frame_id:07d}_obj{obj}_mask_shape"]
    packed = shard_data[f"frame{frame_id:07d}_obj{obj}_mask_packed"]
    H, W = shape
    total = H * W
    return np.unpackbits(packed)[:total].reshape(H, W).astype(bool)


def check_point_in_mask(x, y, mask, xs_mask, ys_mask, tolerance_px):
    """Returns (is_valid, distance) for a point relative to its own mask."""
    xi, yi = int(round(x)), int(round(y))
    H, W = mask.shape
    if not (0 <= yi < H and 0 <= xi < W):
        return False, None  # outside the image bounds

    if mask[yi, xi]:
        return True, 0.0

    if len(xs_mask) == 0:
        return None, None  # no mask available for this subject in this frame -> can't judge

    d = np.min(np.hypot(xs_mask - x, ys_mask - y))
    return (d <= tolerance_px), d


def run_cleanup(df, shards_dir, tolerance_px=15):
    df = df.copy()
    df["mask_status"] = "not_checked"      # not_checked / kept / removed
    df["orig_x"] = df["x"]
    df["orig_y"] = df["y"]
    df["orig_likelihood"] = df["likelihood"]

    shards = sorted(glob.glob(str(Path(shards_dir) / "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {shards_dir}")

    to_remove_idx = []
    t0 = time.time()

    for shard_idx, shard_path in enumerate(shards):
        shard_data = np.load(shard_path, allow_pickle=True)
        frame_keys = sorted(set(
            int(k.split("_")[0].replace("frame", ""))
            for k in shard_data.files if "_obj1_mask_shape" in k
        ))

        for frame_id in frame_keys:
            mask1 = decode_mask(shard_data, frame_id, 1)
            mask2 = decode_mask(shard_data, frame_id, 2)
            ys1, xs1 = np.where(mask1)
            ys2, xs2 = np.where(mask2)
            masks = {1: (mask1, xs1, ys1), 2: (mask2, xs2, ys2)}

            frame_idx = df.index[df["frame_id"] == frame_id]
            for idx in frame_idx:
                row = df.loc[idx]
                if pd.isna(row["x"]) or pd.isna(row["y"]):
                    continue
                if row["joint"] in NEVER_DELETE_JOINTS:
                    df.at[idx, "mask_status"] = "kept_protected"
                    continue
                obj_num = SUBJECT_TO_OBJ.get(row["subject_corrected"])
                if obj_num is None:
                    continue
                mask, xs_m, ys_m = masks[obj_num]
                valid, _ = check_point_in_mask(row["x"], row["y"], mask, xs_m, ys_m, tolerance_px)

                if valid is None:
                    continue  # no mask available, can't judge -> leave it alone
                elif valid:
                    df.at[idx, "mask_status"] = "kept"
                else:
                    df.at[idx, "mask_status"] = "removed_pending"
                    to_remove_idx.append(idx)

        elapsed = time.time() - t0
        print(f"Shard {shard_idx + 1}/{len(shards)} completed ({elapsed:.1f}s total)")

    print(f"\nPoints to remove (outside own mask): {len(to_remove_idx)}")

    for idx in to_remove_idx:
        df.at[idx, "x"] = np.nan
        df.at[idx, "y"] = np.nan
        df.at[idx, "likelihood"] = 0.0
        df.at[idx, "mask_status"] = "removed"

    print(f"Points deleted (x,y = NaN, likelihood = 0.0): {len(to_remove_idx)}")
    return df


def build_report(df, output_path):
    checked = df[df["mask_status"] != "not_checked"]
    removed = checked[checked["mask_status"] == "removed"]

    summary_rows = []
    subjects = sorted(df["subject_corrected"].dropna().unique())
    for joint in ALL_JOINTS:
        row = {"joint": joint}
        total_all = 0
        removed_all = 0
        for subj in subjects:
            total = len(checked[(checked["joint"] == joint) & (checked["subject_corrected"] == subj)])
            rem = len(removed[(removed["joint"] == joint) & (removed["subject_corrected"] == subj)])
            pct = (rem / total * 100) if total > 0 else 0.0
            row[f"{subj}_total"] = total
            row[f"{subj}_removed"] = rem
            row[f"{subj}_pct_removed"] = round(pct, 2)
            total_all += total
            removed_all += rem
        row["total_all"] = total_all
        row["removed_all"] = removed_all
        row["pct_removed_all"] = round((removed_all / total_all * 100) if total_all > 0 else 0.0, 2)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("pct_removed_all", ascending=False)

    overall_total = checked.shape[0]
    overall_removed = removed.shape[0]
    overall_row = pd.DataFrame([{
        "joint": "TOTAL",
        "total_all": overall_total,
        "removed_all": overall_removed,
        "pct_removed_all": round((overall_removed / overall_total * 100) if overall_total > 0 else 0.0, 2),
    }])
    summary_df = pd.concat([summary_df, overall_row], ignore_index=True)

    details_df = removed[[
        "frame_id", "subject_corrected", "joint", "orig_x", "orig_y", "orig_likelihood"
    ]].copy()
    details_df.columns = ["frame_id", "subject", "joint", "x_original", "y_original", "likelihood_original"]
    details_df = details_df.sort_values(["frame_id", "subject", "joint"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        details_df.to_excel(writer, sheet_name="Removed Details", index=False)

    print(f"Report saved: {output_path}")
    print(f"\nOverall summary: {overall_removed}/{overall_total} points removed "
          f"({overall_row.iloc[0]['pct_removed_all']}%)")
    print("\nJoints sorted by how prone they are to errors (highest percentage first):")
    print(summary_df[["joint", "total_all", "removed_all", "pct_removed_all"]].head(6).to_string(index=False))


def to_dlc_wide_format(df, scorer_name):
    """
    Reshapes the long-format dataframe (frame_id, subject_corrected, joint, x, y,
    likelihood) into a wide, multi-row-header CSV matching the original DLC layout:
    header levels = scorer / individuals / bodyparts / coords, one row per frame.
    This is the format deepOF expects when loading DLC-style tracking files.

    IMPORTANT: for a given (frame_id, subject_corrected, joint) there can be TWO rows
    in the input — one real detection, and one leftover "nan_coords" row (from the
    other original DLC individual, which happened to keep/fall back to the same
    subject label). If both were passed to pivot_table(aggfunc="first"), pandas
    would pick the first non-null value INDEPENDENTLY per column, which can silently
    combine x/y from one row with likelihood from the other. To prevent that, we
    de-duplicate first, always keeping the row that actually has coordinates (if any)
    so x, y and likelihood always come from the SAME original row.
    """
    long_df = df[["frame_id", "subject_corrected", "joint", "x", "y", "likelihood"]].copy()

    has_coords = long_df["x"].notna() & long_df["y"].notna()
    long_df = long_df.assign(_has_coords=has_coords)
    long_df = long_df.sort_values("_has_coords", ascending=False)
    before = len(long_df)
    long_df = long_df.drop_duplicates(subset=["frame_id", "subject_corrected", "joint"], keep="first")
    n_dupes = before - len(long_df)
    if n_dupes > 0:
        print(f"Note: removed {n_dupes} duplicate (frame, subject, joint) rows before building "
              f"the wide CSV (kept the one with real coordinates where available).")
    long_df = long_df.drop(columns=["_has_coords"])

    long_df = long_df.rename(columns={"subject_corrected": "individuals", "joint": "bodyparts"})
    long_df["bodyparts"] = long_df["bodyparts"].str.title()  # nose -> Nose, right_fhip -> Right_Fhip

    wide = long_df.pivot_table(
        index="frame_id",
        columns=["individuals", "bodyparts"],
        values=["x", "y", "likelihood"],
        aggfunc="first",
    )
    # pivot_table puts the "values" name (x/y/likelihood) as the outermost column level;
    # reorder to (individuals, bodyparts, coords) and add the scorer level on top.
    wide = wide.swaplevel(0, 2, axis=1).swaplevel(0, 1, axis=1)
    wide = wide.sort_index(axis=1, level=[0, 1])
    wide.columns = pd.MultiIndex.from_tuples(
        [(scorer_name, ind, bp, coord) for (ind, bp, coord) in wide.columns],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    wide.index.name = None  # avoid an extra header row in the CSV (real DLC files have exactly 4 header rows)
    return wide


def run(input_csv, shards_dir, output_dir, tolerance_px=15, scorer_name="corrected_tracking"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_csv).stem.replace("_corrected_anatomical", "")

    df = pd.read_csv(input_csv)
    print(f"Total frames: {df['frame_id'].nunique()}, total rows: {len(df)}")

    cleaned = run_cleanup(df, shards_dir, tolerance_px)

    report_path = out_dir / f"{stem}_mask_cleanup_report.xlsx"
    build_report(cleaned, report_path)

    print("\nBuilding final DLC-style wide CSV for deepOF...")
    wide = to_dlc_wide_format(cleaned, scorer_name)
    cleaned_path = out_dir / f"{stem}_cleaned.csv"
    wide.to_csv(cleaned_path)
    print(f"Final cleaned CSV (DLC wide format) saved: {cleaned_path}")

    return cleaned_path, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, help="CSV from anatomical_correction.py (*_corrected_anatomical.csv)")
    parser.add_argument("--shards_dir", required=True, help="Folder with shard_*.npz files (SAM3 masks)")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--tolerance_px", type=int, default=15)
    parser.add_argument("--scorer_name", default="corrected_tracking",
                         help="Value to use for the 'scorer' header level in the output CSV")
    args = parser.parse_args()

    run(args.input_csv, args.shards_dir, args.output_dir, args.tolerance_px, args.scorer_name)


if __name__ == "__main__":
    main()
