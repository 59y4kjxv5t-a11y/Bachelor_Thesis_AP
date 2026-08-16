"""
Anatomical correction of anomalous body parts.

Logic (swap only, no interpolation):
    If a detection violates the anatomical threshold, try to swap it with the
    other detection for the same joint (if available and if the swap improves
    both subjects). If no valid swap is found, the point is left unchanged.

NOTE: internal column names (subject_corrected, joint, x, y, likelihood, reason, ...) are kept
as-is on purpose, since downstream scripts (mask_cleanup.py, visualize_corrected.py)
read these exact column names from the CSV. The "likelihood" column (the original DLC
confidence value) is carried through as-is; when a swap happens, it travels with the
point it belongs to.
"""

import numpy as np
import pandas as pd
import argparse

SKELETON_EDGES = [
    ("nose","right_ear"),("nose","left_ear"),
    ("right_ear","spine_1"),("left_ear","spine_1"),
    ("nose","spine_1"),("spine_1","center"),
    ("center","spine_2"),("spine_2","tail_base"),
    ("tail_base","tail_1"),("tail_1","tail_2"),
    ("tail_2","tail_tip"),("spine_1","right_fhip"),
    ("spine_1","left_fhip"),("spine_2","right_bhip"),
    ("spine_2","left_bhip"),
]

THRESHOLDS = {
    ("nose","right_ear"):   117.0,
    ("nose","left_ear"):    110.0,
    ("right_ear","spine_1"): 44.6,
    ("left_ear","spine_1"):  49.0,
    ("nose","spine_1"):     124.4,
    ("spine_1","center"):    96.8,
    ("center","spine_2"):    78.0,
    ("spine_2","tail_base"): 60.6,
    ("tail_base","tail_1"):  97.0,
    ("tail_1","tail_2"):    102.4,
    ("tail_2","tail_tip"):  111.8,
    ("spine_1","right_fhip"): 56.0,
    ("spine_1","left_fhip"):  59.2,
    ("spine_2","right_bhip"): 74.0,
    ("spine_2","left_bhip"):  77.2,
}

JOINT_NEIGHBORS = {}
for (j1, j2), thresh in THRESHOLDS.items():
    JOINT_NEIGHBORS.setdefault(j1, []).append((j2, thresh))
    JOINT_NEIGHBORS.setdefault(j2, []).append((j1, thresh))


def get_pts(frame_df, subj):
    sub = frame_df[frame_df["subject_corrected"]==subj].dropna(subset=["x","y"])
    return {r["joint"]: (r["x"], r["y"], r["likelihood"]) for _, r in sub.iterrows()}


def is_valid(joint, x, y, pts, factor=1.0):
    for neighbor, thresh in JOINT_NEIGHBORS.get(joint, []):
        if neighbor not in pts:
            continue
        if np.hypot(x - pts[neighbor][0], y - pts[neighbor][1]) > thresh * factor:
            return False
    return True


def correct_frame(frame_df, subjects, frame_id):
    frame_df = frame_df.copy()
    subj1, subj2 = subjects

    changed = True
    max_iter = 10
    iteration = 0

    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        pts = {s: get_pts(frame_df, s) for s in subjects}

        for subj in subjects:
            other = subj2 if subj == subj1 else subj1

            for joint in list(pts[subj].keys()):
                x, y, lh = pts[subj][joint]
                if is_valid(joint, x, y, pts[subj]):
                    continue

                # --- Attempt: swap with the other subject's detection ---
                if joint in pts[other]:
                    ox, oy, olh = pts[other][joint]
                    if (is_valid(joint, ox, oy, pts[subj]) and
                            is_valid(joint, x, y, pts[other])):
                        ms = (frame_df["subject_corrected"]==subj)  & (frame_df["joint"]==joint)
                        mo = (frame_df["subject_corrected"]==other) & (frame_df["joint"]==joint)
                        frame_df.loc[ms, ["x","y","likelihood"]] = [ox, oy, olh]
                        frame_df.loc[mo, ["x","y","likelihood"]] = [x,  y,  lh]
                        frame_df.loc[ms, "reason"] = "anat_swap"
                        frame_df.loc[mo, "reason"] = "anat_swap"
                        pts = {s: get_pts(frame_df, s) for s in subjects}
                        changed = True
                        break  # restart the outer loop
                # If no valid swap is found, the point is left unchanged (no interpolation).

    return frame_df


def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    subjects = sorted(df["subject_corrected"].dropna().unique())[:2]
    print(f"Subjects: {subjects}")
    print(f"Total frames: {df['frame_id'].nunique()}")

    corrected = []
    n_swap = 0

    for i, (frame_id, group) in enumerate(df.groupby("frame_id")):
        fixed = correct_frame(group, subjects, frame_id)
        n_swap += (fixed["reason"] == "anat_swap").sum()
        corrected.append(fixed)
        if i % 2000 == 0:
            print(f"  frame {frame_id}: swap={n_swap}")

    out = pd.concat(corrected, ignore_index=True)
    out.to_csv(output_csv, index=False)
    print(f"\nDone — swaps: {n_swap}")
    print(f"Saved: {output_csv}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    run(args.input_csv, args.output_csv)
