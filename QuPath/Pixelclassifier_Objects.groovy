// ============================================================
// Batch: Apply Pixel Classifier + Create Objects
// ============================================================

import qupath.lib.scripting.QP
import qupath.lib.objects.classes.PathClass

def project = getProject()

project.getImageList().each { entry ->
    def imageName = entry.getImageName()
    
    if (!imageName.contains('cFOSseg') && !imageName.contains('GFPseg')) return
    
    print 'Processing: ' + imageName
    
    def imageData = entry.readImageData()
    setBatchProjectAndImage(project, imageData)
    
    def classifierName = imageName.contains('cFOSseg') ? 'pixelclass_cFOS' : 'pixelclass_GFP'
    def minArea = imageName.contains('cFOSseg') ? 15.0 : 40.0
    
    def classifier = project.getPixelClassifiers().get(classifierName)
    
    if (classifier == null) {
        print 'Classifier not found: ' + classifierName + ' – skipped'
        return
    }

    // Use first annotation as region
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    
    if (annotations.isEmpty()) {
        print 'No annotation found in: ' + imageName + ' – skipped'
        return
    }

    // Set annotation as selected so classifier runs only within it
    hierarchy.getSelectionModel().setSelectedObject(annotations[0])
    
    PixelClassifierTools.createAnnotationsFromPixelClassifier(
        imageData, classifier, minArea, 0.0,
        PixelClassifierTools.CreateObjectOptions.SPLIT,
        PixelClassifierTools.CreateObjectOptions.DELETE_EXISTING
    )
    
    entry.saveImageData(imageData)
    print 'Done: ' + imageName
}

print '================================'
print 'All images processed!'
print '================================'