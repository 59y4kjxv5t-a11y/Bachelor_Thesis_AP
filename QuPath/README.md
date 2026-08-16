# QuPath Scripts

Groovy scripts for cell detection and colocalization analysis in QuPath.

1. **Export ROI** – exports manually drawn ROIs to a folder
2. **Import ROI** – imports the ROIs onto new (e.g. Ilastik-segmented) images
3. **Pixel Classifier + Objects** – detects cells using the trained GFP/cFos pixel classifiers (`.json` classifier files included) + creates objects 
4. **Export Objects** – exports the detected cell shapes
5. **Batch Colocalization + Visualization** – computes GFP/cFos overlap and visualizes it on the original image
