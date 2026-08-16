// ============================================================
// Import ROIs from GeoJSON onto segmentation images
// Matches by image name prefix (e.g. count3_1)
// ============================================================

import qupath.lib.io.PathIO
import javax.swing.JFileChooser

// Select ROI folder
def chooser = new JFileChooser()
chooser.setDialogTitle("Select GeoJSON ROI folder")
chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY)
if (chooser.showOpenDialog(null) != JFileChooser.APPROVE_OPTION) { print 'Cancelled.'; return }
def roiDir = chooser.getSelectedFile().getAbsolutePath()

getProject().getImageList().each { entry ->
    def imageName = entry.getImageName()
    
    // Only segmentation images
    if (!imageName.contains('cFOSseg') && !imageName.contains('GFPseg')) return
    
    // Extract prefix e.g. count3_1
    def prefix = imageName
        .replaceAll('_cFOSseg.*', '')
        .replaceAll('_GFPseg.*', '')
        .replaceAll('\\.tif.*', '')
    
    // Find matching GeoJSON
    def roiFile = new File(roiDir + File.separator + prefix + '.geojson')
    
    if (!roiFile.exists()) {
        print 'No ROI found for: ' + prefix + ' – skipped'
        return
    }
    
    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()
    
    // Clear existing annotations
    hierarchy.getAnnotationObjects().clear()
    hierarchy.fireHierarchyChangedEvent(this)
    
    // Import ROI
    def roiObjects = PathIO.readObjects(roiFile)
    hierarchy.addObjects(roiObjects)
    hierarchy.fireHierarchyChangedEvent(this)
    
    entry.saveImageData(imageData)
    print 'ROI imported: ' + imageName
}

print '================================'
print 'Done!'
print '================================'