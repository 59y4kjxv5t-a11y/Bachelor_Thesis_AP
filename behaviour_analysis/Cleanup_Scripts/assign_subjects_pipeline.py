"""
Subject-identity correction pipeline based on SAM3 masks.

For every keypoint predicted by DeepLabCut, reassigns the correct subject (exp/stim)
using the SAM3 segmentation masks (one per object) as reference, with a mutual
exclusion constraint (one subject per mask, per joint).

Logic: geometry (distance to mask/centroid) + constrained optimal assignment,
no training, no neural network involved.

The original DLC "likelihood" value is carried through unchanged for every point
(it is NOT recomputed here) so downstream steps and the final deepOF-ready file
keep the real per-point confidence values.

USAGE:
    python assign_subjects_pipeline.py \
        --pred_csv path/to/dlc_predicted_csv.csv \
        --shards_dir path/to/folder/with/shard_*.npz \
        --centroids path/to/centroids.npz \
        --output_dir path/to/output/folder

OUTPUT (inside the output folder):
    - <name>_assignment_raw.csv      : per-keypoint assignment, long format
                                        (intermediate file, used as input for step 2)

NOTE: internal column names (subject, subject_corrected, pred_subject_original, joint,
likelihood, ...) are kept as-is on purpose, since downstream scripts
(anatomical_correction.py, mask_cleanup.py, visualize_corrected.py) read these exact
column names from the CSV.
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============ DLC CSV CONVERSION FUNCTIONS (wide -> long) ============

def convert_dlc_predicted_to_long(csv_path):
    """Converts a DLC OUTPUT CSV (predictions, with likelihood) from wide to long format."""
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3])
    frame_id = df.iloc[:, 0].astype(int)
    data_cols = df.columns[1:]
    data = df.iloc[:, 1:].copy()
    data.columns = pd.MultiIndex.from_tuples(data_cols, names=["scorer", "subject", "joint", "coord"])
    data.insert(0, "frame_id", frame_id.values)
    long_df = data.set_index("frame_id").stack(level=["subject", "joint"], future_stack=True).reset_index()
    long_df.columns = ["frame_id", "subject", "joint", "x", "y", "likelihood"]
    return long_df


# ============ SAM3 MASK FUNCTIONS ============

def decode_mask(shard_data, frame_id, obj):
    shape = shard_data[f"frame{frame_id:07d}_obj{obj}_mask_shape"]
    packed = shard_data[f"frame{frame_id:07d}_obj{obj}_mask_packed"]
    H, W = shape
    total = H * W
    return np.unpackbits(packed)[:total].reshape(H, W).astype(bool)


def assign_by_mask(x, y, mask1, mask2, xs1, ys1, xs2, ys2, tolerance_px=15):
    xi, yi = int(round(x)), int(round(y))
    H, W = mask1.shape
    if not (0 <= yi < H and 0 <= xi < W):
        return None, "out_of_bounds"

    in1 = mask1[yi, xi]
    in2 = mask2[yi, xi]

    if in1 and not in2:
        return "obj1", "mask_unique"
    elif in2 and not in1:
        return "obj2", "mask_unique"
    elif in1 and in2:
        return None, "overlap"

    d1 = np.min(np.hypot(xs1 - x, ys1 - y)) if len(xs1) > 0 else np.inf
    d2 = np.min(np.hypot(xs2 - x, ys2 - y)) if len(xs2) > 0 else np.inf

    if min(d1, d2) > tolerance_px:
        return None, "no_mask"
    return ("obj1" if d1 < d2 else "obj2"), "mask_nearest"


# ============ MAIN PIPELINE ============

def run_pipeline(pred_df, shards_dir, centroids_path, tolerance_px=15):
    centroids = np.load(centroids_path, allow_pickle=True)
    c1 = centroids["centroids_obj1"]
    c2 = centroids["centroids_obj2"]

    shards = sorted(glob.glob(str(Path(shards_dir) / "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {shards_dir}")

    all_results = []
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

            frame_kpts = pred_df[pred_df["frame_id"] == frame_id]
            if frame_id >= len(c1) or frame_id >= len(c2):
                continue
            cx1, cy1 = c1[frame_id, 1], c1[frame_id, 2]
            cx2, cy2 = c2[frame_id, 1], c2[frame_id, 2]

            per_joint = {}
            for _, kp in frame_kpts.iterrows():
                per_joint.setdefault(kp["joint"], []).append(
                    (kp["subject"], kp["x"], kp["y"], kp["likelihood"])
                )

            # Dictionary updated in real time with this frame's assignments so far
            assigned_in_frame = {"obj1": {}, "obj2": {}}  # obj -> joint -> (x,y)

            ANATOMICAL_NEIGHBORS = {
                "tail_2":   [("tail_1", 102.4), ("tail_tip", 111.8)],
                "tail_tip": [("tail_2", 111.8)],
                "tail_1":   [("tail_base", 97.0), ("tail_2", 102.4)],
                "nose":     [("spine_1", 124.4), ("right_ear", 117.0), ("left_ear", 110.0)],
            }

            def anat_cost(x, y, obj, joint):
                cost = 0.0
                for neighbor, thresh in ANATOMICAL_NEIGHBORS.get(joint, []):
                    if neighbor in assigned_in_frame[obj]:
                        nx, ny = assigned_in_frame[obj][neighbor]
                        d = np.hypot(x - nx, y - ny)
                        if d > thresh:
                            cost += (d - thresh)
                return cost

            for joint, dets in per_joint.items():
                valid_dets = [(s, x, y, lh) for (s, x, y, lh) in dets if not (pd.isna(x) or pd.isna(y))]
                nan_dets = [(s, x, y, lh) for (s, x, y, lh) in dets if pd.isna(x) or pd.isna(y)]

                for s, x, y, lh in nan_dets:
                    all_results.append((frame_id, joint, s, x, y, lh, None, "nan_coords"))

                if len(valid_dets) == 0:
                    continue

                if len(valid_dets) == 1:
                    s, x, y, lh = valid_dets[0]
                    assign, reason = assign_by_mask(x, y, mask1, mask2, xs1, ys1, xs2, ys2, tolerance_px)
                    if assign is None:
                        d1 = np.hypot(x - cx1, y - cy1)
                        d2 = np.hypot(x - cx2, y - cy2)
                        assign = "obj1" if d1 < d2 else "obj2"
                        reason = (reason or "none") + "_centroid_fallback"
                    assigned_in_frame[assign][joint] = (x, y)
                    all_results.append((frame_id, joint, s, x, y, lh, assign, reason))
                    continue

                # 2 valid detections -> exclusive assignment using mask cost + anatomical cost
                scores = []
                for idx, (s, x, y, lh) in enumerate(valid_dets):
                    d1 = np.min(np.hypot(xs1 - x, ys1 - y)) if len(xs1) > 0 else np.hypot(x - cx1, y - cy1)
                    d2 = np.min(np.hypot(xs2 - x, ys2 - y)) if len(xs2) > 0 else np.hypot(x - cx2, y - cy2)
                    a1 = anat_cost(x, y, "obj1", joint)
                    a2 = anat_cost(x, y, "obj2", joint)
                    scores.append((idx, d1 + a1, d2 + a2))

                cost_a = scores[0][1] + scores[1][2]  # det0->obj1, det1->obj2
                cost_b = scores[0][2] + scores[1][1]  # det0->obj2, det1->obj1
                assign0, assign1 = ("obj1", "obj2") if cost_a <= cost_b else ("obj2", "obj1")

                s0, x0, y0, lh0 = valid_dets[0]
                s1, x1, y1, lh1 = valid_dets[1]
                assigned_in_frame[assign0][joint] = (x0, y0)
                assigned_in_frame[assign1][joint] = (x1, y1)
                all_results.append((frame_id, joint, s0, x0, y0, lh0, assign0, "exclusive_pair"))
                all_results.append((frame_id, joint, s1, x1, y1, lh1, assign1, "exclusive_pair"))

        elapsed = time.time() - t0
        print(f"Shard {shard_idx + 1}/{len(shards)} completed ({elapsed:.1f}s total)")

    result_df = pd.DataFrame(all_results, columns=[
        "frame_id", "joint", "pred_subject_original", "x", "y", "likelihood", "assigned", "reason"
    ])
    return result_df


def map_obj_to_subject_names(result_df, centroids_path):
    """
    Fixed mapping: obj1 = stim, obj2 = exp.
    This rule is fixed across all videos because SAM3 is always started
    with stim as the first object to track.
    """
    mapping = {"obj1": "stim", "obj2": "exp"}
    print(f"Fixed mapping applied: {mapping}")
    result_df["subject_corrected"] = result_df["assigned"].map(mapping)
    result_df.loc[result_df["assigned"].isna(), "subject_corrected"] = \
        result_df.loc[result_df["assigned"].isna(), "pred_subject_original"]
    return result_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_csv", required=True, help="DLC predicted CSV (original wide format)")
    parser.add_argument("--shards_dir", required=True, help="Folder with shard_*.npz files (SAM3 masks)")
    parser.add_argument("--centroids", required=True, help="Path to centroids.npz")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--tolerance_px", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.pred_csv).stem

    print("Converting predicted CSV...")
    pred_df = convert_dlc_predicted_to_long(args.pred_csv)

    print("Running assignment pipeline...")
    result_df = run_pipeline(pred_df, args.shards_dir, args.centroids, args.tolerance_px)

    print("Mapping subject names (fixed: obj1=stim, obj2=exp)...")
    result_df = map_obj_to_subject_names(result_df, args.centroids)

    raw_path = out_dir / f"{stem}_assignment_raw.csv"
    result_df.to_csv(raw_path, index=False)

    print(f"\nDone.")
    print(f"  Assignment (input for step 2): {raw_path}")
    print("\nAssignment reason summary:")
    print(result_df["reason"].value_counts())


if __name__ == "__main__":
    main()
