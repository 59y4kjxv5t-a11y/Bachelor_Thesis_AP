#!/usr/bin/env python
"""
npz_mask_tool.py

Trims a folder of SAM-style mask shards (.npz files) belonging to one video,
using the SAME cutoff frame number you already used to trim the video with
ffmpeg. This keeps the masks perfectly in sync with the trimmed video.

Keys inside each shard look like:
    frame0000377_obj1_mask_packed
    frame0000377_obj1_mask_shape

Workflow:
    1. Give the input folder containing the shard .npz files
    2. Give the cutoff frame number (the LAST frame to keep - same number
       you used when trimming the video)
    3. The script automatically figures out, per shard:
        - if the shard is entirely BEFORE the cutoff -> copy unchanged
        - if the cutoff falls INSIDE the shard -> trim it, keep only
          frames <= cutoff
        - if the shard is entirely AFTER the cutoff -> drop it completely
          (not written to output, since the video doesn't have these
          frames anymore)
    4. All resulting shards are written to the output folder you choose.

Requirements (Anaconda):
    conda install -c conda-forge numpy

Usage:
    python npz_mask_tool.py
"""

import os
import re
import shutil
import sys

import numpy as np

FRAME_KEY_RE = re.compile(r"^frame(\d+)_obj(\d+)_mask_(packed|shape)$")


def ask_path(prompt, must_exist=False, is_folder=False):
    while True:
        path = input(prompt).strip().strip('"')
        if must_exist:
            if is_folder and not os.path.isdir(path):
                print(f"  -> Folder not found: {path}. Please try again.")
                continue
            if not is_folder and not os.path.isfile(path):
                print(f"  -> File not found: {path}. Please try again.")
                continue
        return path


def ask_folder(prompt):
    path = input(prompt).strip().strip('"')
    os.makedirs(path, exist_ok=True)
    return path


def ask_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("  -> Please enter a valid whole number.")


def get_frame_numbers(data):
    frame_numbers = set()
    for key in data.keys():
        m = FRAME_KEY_RE.match(key)
        if m:
            frame_numbers.add(int(m.group(1)))
    return sorted(frame_numbers)


def trim_shard(npz_path, output_path, cutoff_frame):
    data = np.load(npz_path, allow_pickle=True)

    new_data = {}
    for key in data.keys():
        m = FRAME_KEY_RE.match(key)
        if not m:
            new_data[key] = data[key]
            continue
        frame_num = int(m.group(1))
        if frame_num <= cutoff_frame:
            new_data[key] = data[key]

    np.savez(output_path, **new_data)


def main():
    print("=== NPZ Mask Shard Trim Tool (sync with video cutoff) ===\n")

    input_folder = ask_path(
        "Path to the folder containing the shard .npz files: ",
        must_exist=True,
        is_folder=True,
    )

    output_folder = ask_folder("Folder to save the resulting shards into: ")

    cutoff_frame = ask_int(
        "\nCutoff frame number (the LAST frame to keep - same number you used "
        "to trim the video): "
    )

    shard_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".npz")
    )

    if not shard_files:
        print(f"No .npz files found in {input_folder}")
        sys.exit(1)

    print(f"\nFound {len(shard_files)} shard(s). Processing...\n")

    for fname in shard_files:
        full_path = os.path.join(input_folder, fname)
        data = np.load(full_path, allow_pickle=True)
        frame_numbers = get_frame_numbers(data)

        if not frame_numbers:
            print(f"{fname}: no recognizable frame keys, copying unchanged.")
            shutil.copy2(full_path, os.path.join(output_folder, fname))
            continue

        shard_min = frame_numbers[0]
        shard_max = frame_numbers[-1]
        output_path = os.path.join(output_folder, fname)

        if shard_max <= cutoff_frame:
            # Entire shard is before the cutoff -> keep as-is
            shutil.copy2(full_path, output_path)
            print(f"{fname}: frames {shard_min}-{shard_max} (all <= cutoff) -> copied unchanged.")

        elif shard_min > cutoff_frame:
            # Entire shard is after the cutoff -> drop it, video doesn't have these frames
            print(f"{fname}: frames {shard_min}-{shard_max} (all > cutoff) -> dropped (not written).")

        else:
            # Cutoff falls inside this shard -> trim it
            trim_shard(full_path, output_path, cutoff_frame)
            kept_count = sum(1 for f in frame_numbers if f <= cutoff_frame)
            print(f"{fname}: frames {shard_min}-{shard_max} -> trimmed to {kept_count} frame(s) "
                  f"(kept up to frame {cutoff_frame}).")

    print(f"\nDone! Resulting shards saved to: {output_folder}")


if __name__ == "__main__":
    main()
