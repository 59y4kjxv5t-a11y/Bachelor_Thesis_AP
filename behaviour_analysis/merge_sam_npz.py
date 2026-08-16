"""
merge_sam_npz.py

Merges the raw SAM3 segmentation data (.npz: masks + centroids) from
multiple resume runs into ONE continuous dataset per animal.
Companion to merge_sam_videos.py (which does the same for the rendered
.mp4 videos).

Expected structure per animal folder (example "count_0"):
    count_0/
      count_0_app_shards/                          <- output run (base)
        shard_00000.npz, shard_00001.npz, ...
        centroids.npz
      count_0_resume_app_shards/                    <- usually EMPTY (bug)
      count_0_resume_app_shards_resume0000960/       <- resume run, real data
        shard_00000.npz, ...
        centroids.npz
      count_0_resume2_app_shards/                    <- usually EMPTY (bug)
      count_0_resume2_app_shards_resume0016965/       <- resume2 run, real data
        shard_00000.npz, ...
        centroids.npz

npz structure (confirmed by inspection):
  - shard_XXXXX.npz: keys like 'frame0000960_obj1_mask_packed' (bit-packed
    mask) and 'frame0000960_obj1_mask_shape' (original shape, e.g. [720,1280]).
    The frame number in the key is GLOBAL (relative to the original video).
  - centroids.npz: keys like 'centroids_obj1' -> array (N, 3), column 0 =
    global frame number, column 1/2 = x/y. Also globally numbered.

Merge logic (same as for the videos):
  - For each frame, the run with the highest priority that actually has
    data for that frame wins (output < resume < resume2 < resume3 ...).
  - Frame ownership is NOT derived from the folder name, but read directly
    from the keys/rows actually present - more robust than for the videos.
  - Centroids are taken 1:1 from the winning run, not recomputed.
  - Frames not present in ANY run remain a gap (unlike video, cannot be
    filled with a placeholder) -> only a warning, no abort.

Output: per animal a subfolder "<animal>_merged_shards/" with
  shard_00000.npz, shard_00001.npz, ... (FRAMES_PER_OUTPUT_SHARD frames each)
  and one merged centroids.npz

Requires: pip install numpy
"""

import os
import re
import glob
import numpy as np

# =====================================================================
# CONFIGURATION - ADJUST HERE
# =====================================================================

ANIMAL_FOLDERS = [
    r"User\Path\To\Folder\count_2",
	r"User\Path\To\Folder\count_5",
	r"User\Path\To\Folder\count_6"
]

OUTPUT_DIR = r"User\Path\To\Folder\SAM_merge"

# How many frames per output shard file (saves memory for long videos)
FRAMES_PER_OUTPUT_SHARD = 500

# =====================================================================
# END CONFIGURATION
# =====================================================================

SHARDS_MARKER = "_app_shards"
FRAME_KEY_RE = re.compile(r"^frame(\d+)_obj(\d+)_mask_(packed|shape)$")
CENTROID_KEY_RE = re.compile(r"^centroids_obj(\d+)$")


def natural_key(path):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def parse_shards_folder(name):
    """
    'count_0_resume2_app_shards_resume0016965' ->
        prefix='count_0', run_name='resume2'
    'count_0_app_shards' -> prefix='count_0', run_name='output'
    'count_0_resume_app_shards' (empty bug folder) -> run_name='resume'
        (still returned, in doubt simply contributes nothing)
    """
    idx = name.find(SHARDS_MARKER)
    if idx == -1:
        return None
    left = name[:idx]
    m = re.search(r"_resume(\d*)$", left)
    if m:
        prefix = left[: m.start()]
        num = m.group(1)
        run_name = "resume" if num == "" else f"resume{num}"
    else:
        prefix = left
        run_name = "output"
    return prefix, run_name


def run_priority(run_name):
    if run_name == "output":
        return 0
    m = re.fullmatch(r"resume(\d*)", run_name)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def find_run_folders(animal_dir):
    """run_name -> list of folder paths (can be 1 or 2, e.g. empty bug folder + real one)"""
    runs = {}
    for entry in sorted(os.listdir(animal_dir)):
        full = os.path.join(animal_dir, entry)
        if not os.path.isdir(full):
            continue
        parsed = parse_shards_folder(entry)
        if parsed is None:
            continue
        _, run_name = parsed
        runs.setdefault(run_name, []).append(full)
    return runs


def index_masks(folders):
    """
    Scans all shard_*.npz files in the given folders (without decompressing
    the arrays, only the key list). Returns:
      frame_to_path: {frame_num: shard_file_path}
      frame_to_objs: {frame_num: sorted list of obj numbers}
    """
    frame_to_path = {}
    frame_to_objs = {}
    for folder in folders:
        shard_files = sorted(glob.glob(os.path.join(folder, "shard_*.npz")), key=natural_key)
        for shard_path in shard_files:
            with np.load(shard_path) as npz:
                for key in npz.files:
                    m = FRAME_KEY_RE.match(key)
                    if not m:
                        continue
                    frame_num = int(m.group(1))
                    obj_num = int(m.group(2))
                    frame_to_path[frame_num] = shard_path
                    frame_to_objs.setdefault(frame_num, set()).add(obj_num)
    return frame_to_path, frame_to_objs


def index_centroids(folders):
    """
    Loads centroids.npz from the given folders (takes the first one found,
    warns if several exist and contain data).
    Returns: {obj_num: {frame_num: (x, y)}}
    """
    result = {}
    found_any = False
    for folder in folders:
        path = os.path.join(folder, "centroids.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as npz:
            for key in npz.files:
                m = CENTROID_KEY_RE.match(key)
                if not m:
                    continue
                obj_num = int(m.group(1))
                arr = npz[key]
                if arr.shape[0] == 0:
                    continue
                found_any = True
                d = result.setdefault(obj_num, {})
                for row in arr:
                    frame_num = int(round(row[0]))
                    d[frame_num] = (row[1], row[2])
    return result if found_any else {}


def process_animal(animal_dir):
    animal_name = os.path.basename(os.path.normpath(animal_dir))
    print(f"\n=== Processing (npz): {animal_name} ({animal_dir}) ===")

    run_folders = find_run_folders(animal_dir)
    if "output" not in run_folders:
        print("  [ERROR] No '*_app_shards' base folder (output) found -> skipped.")
        return
    print(f"  Runs found: {sorted(run_folders, key=run_priority)}")

    runs_data = []  # list of dict: name, priority, frame_to_path, frame_to_objs, centroids
    for run_name, folders in run_folders.items():
        prio = run_priority(run_name)
        if prio is None:
            continue
        frame_to_path, frame_to_objs = index_masks(folders)
        if not frame_to_path:
            print(f"  [WARNING] Run '{run_name}': no mask data found in {folders} -> skipped.")
            continue
        centroids = index_centroids(folders)
        if not centroids:
            print(f"  [WARNING] Run '{run_name}': no centroids.npz with data found.")
        print(f"  Run '{run_name}': {len(frame_to_path)} frame(s) with mask data "
              f"(frame {min(frame_to_path)}-{max(frame_to_path)})")
        runs_data.append({
            "name": run_name, "priority": prio,
            "frame_to_path": frame_to_path, "frame_to_objs": frame_to_objs,
            "centroids": centroids,
        })

    if not runs_data:
        print("  [ERROR] No usable runs -> skipped.")
        return

    runs_data.sort(key=lambda r: r["priority"])  # low priority first -> higher one overwrites

    # Global frame -> winning run mapping
    final_source = {}   # frame_num -> run_dict
    final_objs = {}      # frame_num -> set of obj_nums (from the winning run)
    for r in runs_data:
        for frame_num, objs in r["frame_to_objs"].items():
            final_source[frame_num] = r
            final_objs[frame_num] = objs

    all_frames = sorted(final_source.keys())
    total_span = all_frames[-1] - all_frames[0] + 1
    n_missing_in_span = total_span - len(all_frames)
    print(f"  Total frames with data: {len(all_frames)} "
          f"(range {all_frames[0]}-{all_frames[-1]})")
    if n_missing_in_span > 0:
        print(f"  [WARNING] {n_missing_in_span} frame(s) within this range have "
              f"no mask data in ANY run -> will remain a gap in the output.")

    out_subdir = os.path.join(OUTPUT_DIR, f"{animal_name}_merged_shards")
    os.makedirs(out_subdir, exist_ok=True)

    # --- Write masks, in chunks ---
    open_cache = {"path": None, "npz": None}

    def get_npz(path):
        if open_cache["path"] != path:
            if open_cache["npz"] is not None:
                open_cache["npz"].close()
            open_cache["npz"] = np.load(path)
            open_cache["path"] = path
        return open_cache["npz"]

    shard_idx = 0
    for chunk_start in range(0, len(all_frames), FRAMES_PER_OUTPUT_SHARD):
        chunk_frames = all_frames[chunk_start: chunk_start + FRAMES_PER_OUTPUT_SHARD]
        out_dict = {}
        for frame_num in chunk_frames:
            r = final_source[frame_num]
            shard_path = r["frame_to_path"][frame_num]
            npz = get_npz(shard_path)
            for obj_num in sorted(final_objs[frame_num]):
                k_packed = f"frame{frame_num:07d}_obj{obj_num}_mask_packed"
                k_shape = f"frame{frame_num:07d}_obj{obj_num}_mask_shape"
                if k_packed in npz.files:
                    out_dict[k_packed] = npz[k_packed]
                    out_dict[k_shape] = npz[k_shape]
        out_path = os.path.join(out_subdir, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(out_path, **out_dict)
        print(f"    written: {out_path} ({len(chunk_frames)} frames)")
        shard_idx += 1

    if open_cache["npz"] is not None:
        open_cache["npz"].close()

    # --- Merge and write centroids ---
    all_obj_nums = set()
    for r in runs_data:
        all_obj_nums.update(r["centroids"].keys())

    centroid_out = {}
    for obj_num in sorted(all_obj_nums):
        rows = []
        for frame_num in all_frames:
            if obj_num not in final_objs.get(frame_num, set()):
                continue
            r = final_source[frame_num]
            c = r["centroids"].get(obj_num, {}).get(frame_num)
            if c is not None:
                rows.append([frame_num, c[0], c[1]])
        if rows:
            centroid_out[f"centroids_obj{obj_num}"] = np.array(rows, dtype="float64")

    if centroid_out:
        centroid_path = os.path.join(out_subdir, "centroids.npz")
        np.savez(centroid_path, **centroid_out)
        print(f"    written: {centroid_path}")
    else:
        print("  [WARNING] No centroid data found to merge.")

    print(f"  Done! Result in: {out_subdir}")


def main():
    for animal_dir in ANIMAL_FOLDERS:
        if not os.path.isdir(animal_dir):
            print(f"[ERROR] Folder does not exist: {animal_dir}")
            continue
        process_animal(animal_dir)


if __name__ == "__main__":
    main()
