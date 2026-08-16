#!/usr/bin/env python
"""
video_frame_tool.py

Interactive tool for trimming one or multiple video "shards".

Two modes:
  1) Single video: give one input video path.
  2) Folder mode: give an input folder containing multiple video shards
     (e.g. count_1.mp4, count_2.mp4, ...). The script processes each one:
       - extracts the last N seconds as labeled frame images
       - asks you which frame to cut at (per shard)
       - trims and saves each shard into the output folder

Requirements (Anaconda):
    conda create --prefix D:\conda_envs\video_tools -c conda-forge python=3.11 opencv ffmpeg -y
    conda activate D:\conda_envs\video_tools

Usage:
    python video_frame_tool.py
"""

import os
import subprocess
import sys

import cv2

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


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


def ask_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  -> Please enter a valid number.")


def ask_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("  -> Please enter a valid whole number.")


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes", "j", "ja"):
            return True
        if answer in ("n", "no", "nein"):
            return False
        print("  -> Please answer y/n.")


def extract_last_frames(input_video, output_folder, seconds_from_end):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: could not open video: {input_video}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("Error: could not detect fps from video.")
        cap.release()
        return None, None

    print(f"\nDetected fps: {fps}")
    print(f"Total frames in video: {total_frames}")

    frames_to_extract = int(round(seconds_from_end * fps))
    start_frame = max(total_frames - frames_to_extract, 0)

    print(f"Extracting frames from frame {start_frame} to {total_frames - 1} "
          f"({total_frames - start_frame} frames)...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_folder, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(filename, frame)
        saved_count += 1
        frame_idx += 1

    cap.release()

    print(f"Saved {saved_count} images to: {output_folder}")
    return fps, total_frames


def trim_video(input_video, output_video_path, end_frame, fps):
    end_time_seconds = end_frame / fps

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"trim=end_frame={end_frame},setpts=PTS-STARTPTS",
        "-af", f"atrim=end={end_time_seconds},asetpts=PTS-STARTPTS",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_video_path,
    ]

    print("\nRunning ffmpeg:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nffmpeg failed for {input_video}. Check the error output above.")
        return False

    print(f"\nDone! Trimmed video saved to: {output_video_path}")
    return True


def process_single_video(input_video, frames_root_folder, output_folder, seconds_from_end):
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    frames_subfolder = os.path.join(frames_root_folder, video_name)
    os.makedirs(frames_subfolder, exist_ok=True)

    print(f"\n===== Processing: {input_video} =====")

    fps, total_frames = extract_last_frames(input_video, frames_subfolder, seconds_from_end)
    if fps is None:
        print(f"Skipping {input_video} due to error.")
        return

    print(f"\nLook through the images in: {frames_subfolder}")
    print("Filenames correspond to the real frame number in the original video.\n")

    end_frame = ask_int(
        f"Which frame number should be the LAST frame kept for '{video_name}' "
        f"(cut after this frame)? Enter -1 to skip this video: "
    )

    if end_frame == -1:
        print(f"Skipping trim for {video_name}.")
        return

    output_filename = f"{video_name}_trim.mp4"
    output_video_path = os.path.join(output_folder, output_filename)

    trim_video(input_video, output_video_path, end_frame, fps)


def main():
    print("=== Video Frame Inspection & Trim Tool ===\n")

    folder_mode = ask_yes_no(
        "Do you want to process a whole FOLDER of video shards? (y/n): "
    )

    frames_root_folder = ask_folder(
        "Root folder to save extracted frame images into (subfolders per video will be created): "
    )
    seconds_from_end = ask_float("How many seconds from the end do you want to inspect (per video)? ")
    output_folder = ask_folder("Folder to save the trimmed video(s): ")

    if folder_mode:
        input_folder = ask_path(
            "Path to input folder (can be the folder with shards directly, "
            "or a parent folder containing one subfolder per video): ",
            must_exist=True,
            is_folder=True,
        )

        video_files = sorted(
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        )

        # If no video files directly in this folder, check for subfolders
        # (e.g. parent_folder/video1/, parent_folder/video2/, ...)
        if not video_files:
            subfolders = sorted(
                f for f in os.listdir(input_folder)
                if os.path.isdir(os.path.join(input_folder, f))
            )

            if not subfolders:
                print(f"No video files or subfolders found in {input_folder} "
                      f"(looked for extensions: {', '.join(VIDEO_EXTENSIONS)})")
                sys.exit(1)

            print(f"\nNo videos directly in '{input_folder}', but found these subfolders:")
            for idx, folder_name in enumerate(subfolders, start=1):
                print(f"  {idx}. {folder_name}")

            choice = ask_int(
                "\nWhich video (subfolder) do you want to process? Enter the number: "
            )
            if choice < 1 or choice > len(subfolders):
                print("Invalid choice.")
                sys.exit(1)

            chosen_subfolder = os.path.join(input_folder, subfolders[choice - 1])

            video_files = sorted(
                os.path.join(chosen_subfolder, f)
                for f in os.listdir(chosen_subfolder)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            )

            if not video_files:
                print(f"No video files found in {chosen_subfolder} "
                      f"(looked for extensions: {', '.join(VIDEO_EXTENSIONS)})")
                sys.exit(1)

        print(f"\nFound {len(video_files)} video(s):")
        for idx, v in enumerate(video_files, start=1):
            print(f"  {idx}. {v}")

        selection = input(
            "\nWhich video(s) do you want to process? "
            "Enter numbers separated by commas (e.g. 1,3), a range (e.g. 1-3), "
            "or 'all' for everything: "
        ).strip().lower()

        if selection == "all":
            selected_files = video_files
        else:
            selected_indices = set()
            for part in selection.split(","):
                part = part.strip()
                if "-" in part:
                    start_s, end_s = part.split("-")
                    selected_indices.update(range(int(start_s), int(end_s) + 1))
                elif part:
                    selected_indices.add(int(part))

            selected_files = [
                video_files[i - 1] for i in sorted(selected_indices)
                if 1 <= i <= len(video_files)
            ]

        if not selected_files:
            print("No valid videos selected.")
            sys.exit(1)

        print(f"\nSelected {len(selected_files)} video(s) to process:")
        for v in selected_files:
            print(f"  - {v}")

        for video_path in selected_files:
            process_single_video(video_path, frames_root_folder, output_folder, seconds_from_end)

        print("\nSelected videos processed.")

    else:
        input_video = ask_path("Path to input video: ", must_exist=True)
        process_single_video(input_video, frames_root_folder, output_folder, seconds_from_end)


if __name__ == "__main__":
    main()
