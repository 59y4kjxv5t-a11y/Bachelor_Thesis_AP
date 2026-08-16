"""
Generates a verification video with the corrected body parts (points only, no skeleton
unless --skeleton is passed).
Colors: cyan = subject 1 (default: stim), yellow = subject 2 (default: exp).

USAGE:
    python visualize_corrected.py \
        --assignment_csv  path/to/*_cleaned.csv \
        --video_in        path/to/original_video.mp4 \
        --video_out       path/to/output.mp4 \
        [--segments       "0-300,7100-8900"]   (default: whole video)
        [--subj1          stim]                 (name of subject 1, drawn in cyan)
        [--subj2          exp]                  (name of subject 2, drawn in yellow)

NOTE: reads the wide, multi-row-header DLC-style CSV produced by mask_cleanup.py
(header levels: scorer / individuals / bodyparts / coords), i.e. the same file
used as the deepOF tracking input.
"""

import argparse
import cv2
import pandas as pd
import numpy as np
from pathlib import Path


COLORS = {0: (255, 255, 0), 1: (0, 255, 255)}  # BGR: cyan, yellow

SKELETON_EDGES = [
    ("nose", "right_ear"),
    ("nose", "left_ear"),
    ("right_ear", "spine_1"),
    ("left_ear", "spine_1"),
    ("nose", "spine_1"),
    ("spine_1", "center"),
    ("center", "spine_2"),
    ("spine_2", "tail_base"),
    ("tail_base", "tail_1"),
    ("tail_1", "tail_2"),
    ("tail_2", "tail_tip"),
    ("spine_1", "right_fhip"),
    ("spine_1", "left_fhip"),
    ("spine_2", "right_bhip"),
    ("spine_2", "left_bhip"),
]


def parse_segments(s, total_frames):
    """Converts a string like "0-300,7100-8900" into a list of tuples, or the whole video."""
    if s is None:
        return [(0, total_frames)]
    segments = []
    for part in s.split(","):
        a, b = part.strip().split("-")
        segments.append((int(a), int(b)))
    return segments


def load_wide_dlc_csv(path):
    """
    Reads the wide, 4-header-row DLC-style CSV (scorer / individuals / bodyparts / coords)
    and returns (df, scorer_name, subjects, joints_per_subject).
    """
    df = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
    df.index = df.index.astype(int)
    scorer_name = df.columns.get_level_values("scorer")[0]
    subjects = sorted(df.columns.get_level_values("individuals").unique())
    joints_per_subject = {
        subj: sorted(df[scorer_name][subj].columns.get_level_values("bodyparts").unique())
        for subj in subjects
    }
    return df, scorer_name, subjects, joints_per_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment_csv", required=True,
                        help="Wide DLC-style CSV produced by mask_cleanup.py (*_cleaned.csv)")
    parser.add_argument("--video_in", required=True, help="Original input video")
    parser.add_argument("--video_out", required=True, help="Output video with the points drawn on it")
    parser.add_argument("--segments", default=None,
                        help="Segments to include, e.g. '0-300,7100-8900'. Default: whole video.")
    parser.add_argument("--subj1", default=None,
                        help="Name of subject 1 (cyan). Default: first in alphabetical order.")
    parser.add_argument("--subj2", default=None,
                        help="Name of subject 2 (yellow). Default: second in alphabetical order.")
    parser.add_argument("--point_radius", type=int, default=5, help="Point radius in pixels (default 5)")
    parser.add_argument("--skeleton", action="store_true", help="If set, also draw the skeleton lines")
    args = parser.parse_args()

    print("Loading data...")
    df, scorer_name, subjects, joints_per_subject = load_wide_dlc_csv(args.assignment_csv)

    subj1 = args.subj1 if args.subj1 else subjects[0]
    subj2 = args.subj2 if args.subj2 else subjects[1] if len(subjects) > 1 else subjects[0]
    subj_color = {subj1: COLORS[0], subj2: COLORS[1]}
    print(f"Subjects: {subj1} (cyan), {subj2} (yellow)")

    cap = cv2.VideoCapture(args.video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} @ {fps:.1f}fps, {total_frames} total frames")

    segments = parse_segments(args.segments, total_frames)
    print(f"Segments: {segments}")

    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.video_out, fourcc, fps, (W, H))

    total_written = 0
    for start, end in segments:
        end = min(end, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_id in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id in df.index:
                row = df.loc[frame_id]
                for subj in (subj1, subj2):
                    if subj not in joints_per_subject:
                        continue
                    color = subj_color.get(subj, (200, 200, 200))
                    pts = {}
                    for joint in joints_per_subject[subj]:
                        x = row[(scorer_name, subj, joint, "x")]
                        y = row[(scorer_name, subj, joint, "y")]
                        if pd.isna(x) or pd.isna(y):
                            continue
                        pts[joint] = (int(x), int(y))

                    if args.skeleton:
                        pts_lower = {k.lower(): v for k, v in pts.items()}
                        for j1, j2 in SKELETON_EDGES:
                            j1, j2 = j1.lower(), j2.lower()
                            if j1 in pts_lower and j2 in pts_lower:
                                cv2.line(frame, pts_lower[j1], pts_lower[j2], color, 2)
                    for j, (x, y) in pts.items():
                        cv2.circle(frame, (x, y), args.point_radius, color, -1)

            cv2.putText(frame, f"frame {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out.write(frame)
            total_written += 1

        print(f"Segment {start}-{end} completed.")

    cap.release()
    out.release()
    print(f"\nVideo saved: {args.video_out}  ({total_written} frames)")


if __name__ == "__main__":
    main()
