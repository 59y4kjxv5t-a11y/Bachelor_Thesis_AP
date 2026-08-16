# IPACL Neuron Counting & Analysis Pipeline

Image analysis and statistical pipeline developed for a B.Sc. thesis investigating whether the same or different IPACL neurons are activated during social interactions of C57BL/6J (BL6) and CD1 mice. The pipeline combines the vGATE system (GFP) with cFOS immunostaining to identify and quantify co-activated neuron populations, and compares automated counts against manual counts.

This repository covers the **automated** parts of the pipeline only (image analysis and behavioural analysis). Manual counting is not included.

## Overview

The workflow processes fluorescence microscopy images of 50 µm brain sections through the following stages:

1. **Fiji** – Z-projection and channel merging
2. **Ilastik** – Pixel classification (separate trained models for GFP and cFOS cells)
3. **QuPath** – Batch application of pixel classifiers, ROI export/import (GeoJSON), and colocalization analysis (nearest-neighbor matching)
4. **R** – Comparison of automated vs. manual counts and downstream statistical visualization

Behavioural analysis (social interaction scoring) is handled separately:

5. **SAM3** – Segmentation of animals in behavioral video
6. **DeepLabCut** – Pose estimation
7. **DeepOF** – Social behavior classification from pose data

## Repository Structure

```
├── fiji_macros/          # Z-projection + channel merge macro
├── ilastik/              # Trained pixel classifiers (GFP, cFOS)
├── qupath/               # Groovy scripts for batch classification, ROI I/O, colocalization
├── behavior_analysis/    # SAM3, cleanup scripts 
├── r_analysis/           # R scripts for cell-count data and behaviour
└── docs/                 # workflows
```

## Documentation

- [`docs/Workflow_Imaging.docx`](docs/Workflow_Imaging.docx) – microscope image acquisition (LAS X)
- [`docs/Workflow_Colocalization.docx`](docs/Workflow_Colocalization.docx) – step-by-step guide through the full image analysis pipeline (Fiji → Ilastik → QuPath), including troubleshooting notes


## Tools
|Image Analysis | Behaviour Analysis |
|---|---|
| Fiji/ImageJ | SAM3 |
| Ilastik | DeepLabCut |
| QuPath | DeepOF |

## Workflow

**Image analysis (neuron counting)**
1. Run the Fiji macro on raw image stacks to generate z-projected, channel-merged images.
2. Classify pixels in Ilastik using the corresponding trained models (GFP or cFOS).
3. In QuPath, apply the classifier via batch script, then run the colocalization script.
4. Import the resulting counts into R and run the analysis.

**Behaviour analysis**
1. Run DeepLabCut to extract pose estimates.
2. Segment animals in raw behavioural video using SAM3.
3. Fix output videos.
4. Combine DeepLabCut body parts with SAM3 segmentation.
5. Run DeepOF on the pose data to classify social behaviors.
6. Import resulting CSV into R and run the analysis.

## AI Assistance

Scripts and code in this repository were developed with the assistance of AI tools (Claude/Anthropic, DeepSeek, Gemini).


## Author

Developed as part of a B.Sc. thesis in Neurobiology

