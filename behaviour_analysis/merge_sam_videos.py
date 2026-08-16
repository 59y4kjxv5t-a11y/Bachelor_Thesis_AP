"""
merge_sam_videos.py

Merges the SAM3 output videos from multiple resume runs into ONE
continuous video. (For the raw .npz shard data there is a separate
script, once the npz structure is known.)

Expected structure per animal folder (example for "count_0"):
    count_0/
      count_0_app_shards/                              <- npz raw data (output run)
      count_0_app_output.mp4                            <- video, output run (frame 0 - end)
      count_0_resume_app_shards/                        <- EMPTY (bug, ignored)
      count_0_resume_app_shards_resume0000960/          <- npz raw data, resume run
      count_0_resume_app_output_resume0000960.mp4       <- video, resume run (from frame 960)
      count_0_resume2_app_shards/                       <- EMPTY (bug, ignored)
      count_0_resume2_app_shards_resume0016965/         <- npz raw data, resume2 run
      count_0_resume2_app_output_resume0016965.mp4      <- video, resume2 run (from frame 16965)

The start frame is read automatically from the file name (the number
after "_resume", e.g. 0000960 -> start at frame 960). A manual text file
is NOT required, but can optionally be added as a cross-check (see
MANUAL_START_FRAMES_FILENAME below) - on mismatch only a warning is
issued, the number found in the file name takes precedence.

Priority on overlap: higher resume index wins (resume2 beats resume beats
output). Gaps (no run covers a frame) are filled from "output"; if output
has nothing there either: black frame + warning.

Requires: pip install opencv-python
"""

import os
import re
import shutil
import cv2

# =====================================================================
# CONFIGURATION - ADJUST HERE
# =====================================================================

# One folder per animal. Add a line for more animals, or leave just one.
ANIMAL_FOLDERS = [
    r"User\Path\To\Folder\count_2",
	r"User\Path\To\Folder\count_5",
	r"User\Path\To\Folder\count_6"
]

# Where the finished merge videos are written (created if needed).
# File is then named e.g. "count_0_merged.mp4"
OUTPUT_DIR = r"User\Path\To\Folder\SAM_merge"

# Optional: name of a manual text file in the animal folder for cross-checking
# the start frames (format: "resume 960" per line). Can be empty/absent.
MANUAL_START_FRAMES_FILENAME = "start_frames.txt"

# =====================================================================
# END CONFIGURATION
# =====================================================================

VIDEO_MARKER = "_app_output"
SHARDS_MARKER = "_app_shards"


def parse_entry(name, marker):
    """
    Splits a file/folder name like
    'count_0_resume2_app_output_resume0016965.mp4' into:
      prefix     -> 'count_0'
      run_name   -> 'output' | 'resume' | 'resume2' | 'resume3' ...
      start_frame-> number from the name (None if absent, e.g. for
                    output or for the empty duplicate folders)
    Returns None if the marker is not present in the name.
    """
    idx = name.find(marker)
    if idx == -1:
        return None

    left = name[:idx]  # e.g. 'count_0', 'count_0_resume', 'count_0_resume2'
    right = name[idx + len(marker):]  # e.g. '.mp4', '_resume0016965.mp4', '_resume0016965', ''

    m = re.search(r"_resume(\d*)$", left)
    if m:
        prefix = left[: m.start()]
        num = m.group(1)
        run_name = "resume" if num == "" else f"resume{num}"
    else:
        prefix = left
        run_name = "output"

    m2 = re.search(r"_resume(\d+)", right)
    start_frame = int(m2.group(1)) if m2 else None

    return prefix, run_name, start_frame


def find_video_runs(animal_dir):
    """Finds all *_app_output*.mp4 videos and assigns them to runs."""
    runs = {}
    for f in sorted(os.listdir(animal_dir)):
        if not f.lower().endswith(".mp4"):
            continue
        parsed = parse_entry(f, VIDEO_MARKER)
        if parsed is None:
            continue
        prefix, run_name, start_frame = parsed
        path = os.path.join(animal_dir, f)
        if run_name == "output":
            start_frame = 0
        if start_frame is None:
            print(f"  [WARNING] '{f}' looks like a resume video, but has no "
                  f"start frame number in the name -> skipped.")
            continue
        runs[run_name] = {"path": path, "start": start_frame, "prefix": prefix}
    return runs


def read_manual_start_frames(animal_dir):
    path = os.path.join(animal_dir, MANUAL_START_FRAMES_FILENAME)
    manual = {}
    if not os.path.exists(path):
        return manual
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) == 2:
                try:
                    manual[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    return manual


def run_priority(run_name):
    if run_name == "output":
        return 0
    m = re.fullmatch(r"resume(\d*)", run_name)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def black_frame(h, w):
    import numpy as np
    return np.zeros((h, w, 3), dtype="uint8")


def process_animal(animal_dir):
    animal_name = os.path.basename(os.path.normpath(animal_dir))
    print(f"\n=== Processing: {animal_name} ({animal_dir}) ===")

    runs = find_video_runs(animal_dir)
    if "output" not in runs:
        print(f"  [ERROR] No '*_app_output.mp4' (base video) found -> skipped.")
        return
    print(f"  Runs found: {sorted(runs, key=run_priority)}")

    # Cross-check with manual file, if present
    manual = read_manual_start_frames(animal_dir)
    for run_name, info in runs.items():
        if run_name in manual and manual[run_name] != info["start"]:
            print(f"  [WARNING] '{run_name}': file name says start frame {info['start']}, "
                  f"{MANUAL_START_FRAMES_FILENAME} says {manual[run_name]}. "
                  f"Using the value from the file name ({info['start']}).")

    runs_info = []
    for run_name, info in runs.items():
        n_frames, fps, w, h = get_video_info(info["path"])
        if n_frames <= 0:
            print(f"  [WARNING] '{run_name}' has 0 readable frames -> skipped.")
            continue
        runs_info.append({
            "name": run_name,
            "path": info["path"],
            "priority": run_priority(run_name),
            "start": info["start"],
            "end": info["start"] + n_frames,
            "n_frames": n_frames,
            "fps": fps, "w": w, "h": h,
        })

    if not runs_info:
        print("  [ERROR] No usable runs -> skipped.")
        return

    base = next((r for r in runs_info if r["name"] == "output"), None)
    if base is None:
        print("  [ERROR] 'output' video was not readable -> skipped.")
        return
    fps, w, h = base["fps"], base["w"], base["h"]

    for r in runs_info:
        if abs(r["fps"] - fps) > 0.1 or r["w"] != w or r["h"] != h:
            print(f"  [WARNING] '{r['name']}' has different fps/resolution "
                  f"({r['fps']:.2f}fps, {r['w']}x{r['h']}) vs. baseline "
                  f"({fps:.2f}fps, {w}x{h}). Will be used anyway, please check!")

    total_len = max(r["end"] for r in runs_info)
    runs_info.sort(key=lambda r: r["priority"])

    source_for_frame = [None] * total_len
    for r in runs_info:
        for i in range(r["start"], min(r["end"], total_len)):
            source_for_frame[i] = r["name"]

    n_gaps = sum(1 for s in source_for_frame if s is None)
    if n_gaps:
        print(f"  [WARNING] {n_gaps} frame(s) not covered by any run -> black inserted.")

    print(f"  Total length: {total_len} frames at {fps:.2f} fps "
          f"({total_len / fps:.1f} seconds)")
    for r in runs_info:
        used = sum(1 for s in source_for_frame if s == r["name"])
        print(f"    '{r['name']}': covers {r['start']}-{r['end']-1}, "
              f"of which actually used: {used} frame(s)")

    caps = {r["name"]: cv2.VideoCapture(r["path"]) for r in runs_info}
    local_pos = {r["name"]: 0 for r in runs_info}
    starts_by_name = {r["name"]: r["start"] for r in runs_info}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{animal_name}_merged.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for i in range(total_len):
        run_name = source_for_frame[i]
        if run_name is None:
            writer.write(black_frame(h, w))
            continue

        cap = caps[run_name]
        target_local = i - starts_by_name[run_name]
        while local_pos[run_name] < target_local:
            cap.grab()
            local_pos[run_name] += 1

        ok, frame = cap.read()
        local_pos[run_name] += 1
        if not ok or frame is None:
            print(f"  [WARNING] Frame {i} not readable from '{run_name}' -> black.")
            frame = black_frame(h, w)
        elif frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)
        if i % 1000 == 0:
            print(f"    ... frame {i}/{total_len}")

    writer.release()
    for cap in caps.values():
        cap.release()

    print(f"  Done! Saved to: {out_path}")


def main():
    for animal_dir in ANIMAL_FOLDERS:
        if not os.path.isdir(animal_dir):
            print(f"[ERROR] Folder does not exist: {animal_dir}")
            continue
        process_animal(animal_dir)


if __name__ == "__main__":
    main()
