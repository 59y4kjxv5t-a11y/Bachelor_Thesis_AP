// ============================================================
// Fiji Macro: Invert and rename segmentation images
// ch01 → cFOSseg
// ch00 → GFPseg
// Example: count3_1_LVCC_RAW_ch01_z_merged.tif → count3_1_cFOSseg.tif
// ============================================================

inputDir  = getDirectory("Select input folder (Ilastik segmentations)");
outputDir = getDirectory("Select output folder");

setBatchMode(true);

list = getFileList(inputDir);
count = 0;

for (i = 0; i < list.length; i++) {
    filename = list[i];
    
    if (!endsWith(filename, ".tif") && !endsWith(filename, ".tiff")) continue;

    open(inputDir + filename);
    
    // Convert to 8-bit
    run("8-bit");
    
    // Stretch to full 0-255 range
    run("Enhance Contrast", "saturated=0 normalize");
    
    // Invert
    run("Invert");
    
    nameWithoutExt = replace(filename, ".tiff", "");
    nameWithoutExt = replace(nameWithoutExt, ".tif", "");
    
    idx = indexOf(nameWithoutExt, "_LVCC");
    if (idx == -1) {
        prefix = nameWithoutExt;
        suffix = "seg";
    } else {
        prefix = substring(nameWithoutExt, 0, idx);
        
        if (indexOf(filename, "ch01") >= 0) {
            suffix = "cFOSseg";
        } else if (indexOf(filename, "ch00") >= 0) {
            suffix = "GFPseg";
        } else {
            suffix = "seg";
        }
    }
    
    newName = prefix + "_" + suffix + ".tif";
    
    saveAs("Tiff", outputDir + newName);
    close();
    count++;
}
setBatchMode(false);

