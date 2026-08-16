// ============================================================
// Script: GeoJSON Export for all images in project
// Exports automatically as count3_1_GFPobjects.geojson etc.
// ============================================================

import qupath.lib.objects.classes.PathClass
import javax.swing.JFileChooser

// Select output folder
def chooser = new JFileChooser()
chooser.setDialogTitle("Select output folder")
chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY)
def result = chooser.showOpenDialog(null)

if (result != JFileChooser.APPROVE_OPTION) {
    print 'Cancelled.'
    return
}

def outputDir = chooser.getSelectedFile().getAbsolutePath()
print 'Saving to: ' + outputDir

getProject().getImageList().each { entry ->
    def imageData = entry.readImageData()
    def imageName = entry.getImageName()
    
    def baseName = imageName
        .replaceAll('_GFPseg.*', '')
        .replaceAll('_cFOSseg.*', '')
        .replaceAll('_LVCC.*', '')
        .replaceAll('\\.tif.*', '')
    
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    
    if (annotations.isEmpty()) {
        print 'No objects in: ' + imageName
        return
    }
    
    def classGFP  = PathClass.fromString('GFP')
    def classCFOS = PathClass.fromString('cFOS')
    
    def gfpObjects  = annotations.findAll { it.getPathClass() == classGFP }
    def cfosObjects = annotations.findAll { it.getPathClass() == classCFOS }
    
    if (!gfpObjects.isEmpty()) {
        def pathGFP = outputDir + File.separator + baseName + '_GFPobjects.geojson'
        new File(pathGFP).withWriter('UTF-8') { writer ->
            writer.write(GsonTools.getInstance(true).toJson(gfpObjects))
        }
        print 'Exported: ' + baseName + '_GFPobjects.geojson'
    }
    
    if (!cfosObjects.isEmpty()) {
        def pathCFOS = outputDir + File.separator + baseName + '_cFOSobjects.geojson'
        new File(pathCFOS).withWriter('UTF-8') { writer ->
            writer.write(GsonTools.getInstance(true).toJson(cfosObjects))
        }
        print 'Exported: ' + baseName + '_cFOSobjects.geojson'
    }
}

print '================================'
print 'Export complete!'
print '================================'