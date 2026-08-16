/**
 * Exports all annotations from ALL images in the currently open project as GeoJSON.
 *
 * Renaming:
 *   Count4_1_LVCC_RAW_final.tif  ->  Count4_1.geojson
 *
 * Usage:
 *   1. Set the target folder below at OUTPUT_DIR.
 *   2. Open the project in QuPath (the images must be present as project entries).
 *   3. Automate -> Show script editor -> paste script -> Run.
 *      (No "Run for project" needed - the script runs through all images by itself.)
 */

import qupath.lib.common.GeneralTools
import static qupath.lib.scripting.QP.*

// ---------------- SETTINGS ----------------
// Set the target folder (on Windows, double the backslashes, e.g. "C:\\Users\\Lu\\Export")
def OUTPUT_DIR = "/Users/annapoll/Desktop/T7_copy/coh2/Count_QuPath_2"

// Suffix to remove from the file name
def SUFFIX_TO_REMOVE = "_LVCC_RAW_final"
// -------------------------------------------------

def dir = new File(OUTPUT_DIR)
if (!dir.exists()) {
    dir.mkdirs()
}

def project = getProject()
if (project == null) {
    print "No project open! Please open a QuPath project containing all images first."
    return
}

def entries = project.getImageList()
print "Starting export for ${entries.size()} images..."

int exported = 0
int skipped = 0

for (entry in entries) {
    def originalName = GeneralTools.getNameWithoutExtension(entry.getImageName())
    def imageData = entry.readImageData()
    def annotations = imageData.getHierarchy().getAnnotationObjects()

    if (annotations.isEmpty()) {
        print "No annotations in '${originalName}' - skipped."
        skipped++
        imageData.getServer().close()
        continue
    }

    // e.g. Count4_1_LVCC_RAW_final -> Count4_1
    def newName = originalName.replaceAll(SUFFIX_TO_REMOVE + '$', "")
    def outputPath = buildFilePath(OUTPUT_DIR, newName + ".geojson")

    exportObjectsToGeoJson(annotations, outputPath, "FEATURE_COLLECTION", "PRETTY_PRINT")
    print "Exported (${annotations.size()} annotations): ${outputPath}"
    exported++

    imageData.getServer().close()
}

print "Done. ${exported} images exported, ${skipped} skipped (no annotations)."
