// Select folders via pop-up window
inputDir = getDirectory("Select the LVCC_RAW folder");
zMergedDir = getDirectory("Select the Z_merged folder");
channelMergedDir = getDirectory("Select the Channel_merged folder");

// Enable batch mode (windows stay hidden)
setBatchMode(true); 

// Start processing all subfolders
processFolder(inputDir);

// Disable batch mode (Fiji shows result)
setBatchMode(false); 

print("--- DONE! All images successfully processed ---");

function processFolder(dir) {
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {
        // If it is a subfolder, go deeper
        if (endsWith(list[i], "/")) {
            processFolder(dir + list[i]);
        } 
        // If it is the GFP image, start processing for this image set
        else if (endsWith(list[i], "ch00.tif")) {
            baseName = replace(list[i], "ch00.tif", "");
            processImageSet(dir, baseName);
        }
    }
}

function processImageSet(dir, base) {
    // Paths to the original 3D Z-stacks in the subfolders
    file_ch00 = dir + base + "ch00.tif"; // GFP
    file_ch01 = dir + base + "ch01.tif"; // cFOS
    file_ch02 = dir + base + "ch02.tif"; // DAPI (optional)

    // Check whether a DAPI channel exists for this image set
    hasDapi = File.exists(file_ch02);

    // ==========================================
    // 1. GFP STEP (Maximum Intensity)
    // ==========================================
    open(file_ch00);
    rename("original_ch00");
    run("Z Project...", "projection=[Max Intensity]");
    rename("GFP_2D");
    saveAs("Tiff", zMergedDir + base + "ch00_z_merged.tif");
    rename("GFP_2D"); 
    close("original_ch00"); 

    // ==========================================
    // 2. cFOS STEP (Maximum Intensity)
    // ==========================================
    open(file_ch01);
    rename("original_ch01");
    run("Z Project...", "projection=[Max Intensity]");
    rename("cFOS_2D");
    saveAs("Tiff", zMergedDir + base + "ch01_z_merged.tif");
    rename("cFOS_2D");
    close("original_ch01");

    // ==========================================
    // 3. DAPI STEP (Maximum Intensity) - nur wenn vorhanden
    // ==========================================
    if (hasDapi) {
        open(file_ch02);
        rename("original_ch02");
        run("Z Project...", "projection=[Max Intensity]");
        rename("DAPI_2D");
        saveAs("Tiff", zMergedDir + base + "ch02_z_merged.tif");
        rename("DAPI_2D");
        close("original_ch02");
    }

    // ==========================================
    // 4. MERGE STEP (Channel fusion)
    // ==========================================
    // c1 (Red) = cFOS, c2 (Green) = GFP, c3 (Blue) = DAPI
    if (hasDapi) {
        run("Merge Channels...", "c1=cFOS_2D c2=GFP_2D c3=DAPI_2D create");
    } else {
        run("Merge Channels...", "c1=cFOS_2D c2=GFP_2D create");
    }
    
    // Save to Channel_merged folder with "final" at the end
    saveAs("Tiff", channelMergedDir + base + "final.tif");
    
    close("*"); // Close finished composite image and clear memory
}
print("--- DONE! All images successfully processed ---");
