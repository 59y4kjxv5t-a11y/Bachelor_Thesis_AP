# Behaviour Analysis

Scripts for animal tracking and subject-ID swap correction.

## Pipeline

`MouseTrack.py` – calls SAM3 (via Hugging Face) to generate segmentation masks for each animal.

`merge_sam_npz.py` and `merge_sam_videos.py` – merge SAM3 output (raw .npz mask/centroid data and rendered .mp4 videos) from multiple resume runs into one continuous dataset per animal, for cases where a run was interrupted and resumed.

`video_frame_tool.py` and `npz_mask_tool.py` – cut merged videos and shard files down to the desired frame range.

Run `Pipeline_swaps_correction.py` (adjust the paths inside first) — it runs the other four scripts in sequence from the same folder:

1. `assign_subjects_pipeline.py`
2. `anatomical_correction.py`
3. `mask_cleanup.py`
4. `visualize_corrected.py` – generates a video for visual QC

Uses the SAM3 masks to correct DeepLabCut subject-ID swaps and outputs a deepOF-ready wide-format DLC CSV.

`TrackFix_CD1.py` – fixes identity swaps between the BL6 and CD-1 animal in the tracking data.

DeepOF is then run on the corrected tracking data to classify social behaviours.

## Attribution

`MouseTrack.py`, `Pipeline_swaps_correction.py`, `assign_subjects_pipeline.py`, `anatomical_correction.py`, `visualize_corrected.py`, and `TrackFix_CD1.py` were originally created by AM (second supervisor), modified by the author. `mask_cleanup.py`, `video_frame_tool.py`, `npz_mask_tool.py`, `merge_sam_npz.py`, and `merge_sam_videos.py` were created by the author.
