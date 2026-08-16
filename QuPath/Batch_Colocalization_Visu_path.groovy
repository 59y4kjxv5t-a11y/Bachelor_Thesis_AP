// ============================================================
// Batch: Import Objects + Colocalization + CSV
// ============================================================

import qupath.lib.io.GsonTools
import qupath.lib.objects.classes.PathClass
import com.google.gson.reflect.TypeToken

def geojsonDir = "/Users/Path/To/Folder"
def csvPath    = "/Users/Path/To/Folder"

double overlapRadius = 10.0

def classGFP    = PathClass.fromString('GFP')
def classCFOS   = PathClass.fromString('cFOS')
def classDouble = PathClass.fromString('GFP: cFOS')

def gson = GsonTools.getInstance()
def type = new TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType()

def csvFile = new File(csvPath)
csvFile.text = 'Image,GFP_only,cFOS_only,Double_positive,Total_GFP,Total_cFOS\n'

getProject().getImageList().each { entry ->
    def imageName = entry.getImageName()
    if (!imageName.contains('LVCC_RAW_final')) return

    def prefix   = imageName.replace('_LVCC_RAW_final.tif', '')
    def gfpFile  = new File(geojsonDir + File.separator + prefix + '_GFPobjects.geojson')
    def cfosFile = new File(geojsonDir + File.separator + prefix + '_cFOSobjects.geojson')

    if (!gfpFile.exists() || !cfosFile.exists()) {
        print 'No GeoJSONs for: ' + prefix + ' – skipped'
        return
    }

    print 'Processing: ' + prefix

    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()
    def pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSizeMicrons()

    hierarchy.clearAll()

    def gfpObjects  = gson.fromJson(gfpFile.text, type) as List
    def cfosObjects = gson.fromJson(cfosFile.text, type) as List

    gfpObjects.each  { it.setPathClass(classGFP) }
    cfosObjects.each { it.setPathClass(classCFOS) }

    hierarchy.addObjects(gfpObjects)
    hierarchy.addObjects(cfosObjects)

    print '  GFP:  ' + gfpObjects.size()
    print '  cFOS: ' + cfosObjects.size()

    // Nearest neighbor matching
    // Each GFP finds closest cFOS within radius
    // Each cFOS can only be matched once
    def matchedCFOS = [] as Set
    def matchedGFP  = [] as Set

    gfpObjects.each { gfp ->
        def gx = gfp.getROI().getCentroidX()
        def gy = gfp.getROI().getCentroidY()

        def bestCFOS = null
        def bestDist = Double.MAX_VALUE

        cfosObjects.each { cfos ->
            if (matchedCFOS.contains(cfos)) return

            def cx = cfos.getROI().getCentroidX()
            def cy = cfos.getROI().getCentroidY()
            double distMicrons = Math.sqrt((gx-cx)**2 + (gy-cy)**2) * pixelSize

            if (distMicrons <= overlapRadius && distMicrons < bestDist) {
                bestDist = distMicrons
                bestCFOS = cfos
            }
        }

        if (bestCFOS != null) {
            matchedGFP.add(gfp)
            matchedCFOS.add(bestCFOS)
        }
    }

    // Set classes
    matchedGFP.each  { it.setPathClass(classDouble) }
    matchedCFOS.each { it.setPathClass(classDouble) }

    def nDouble   = matchedGFP.size()
    def nGFPonly  = gfpObjects.size()  - nDouble
    def nCFOSonly = cfosObjects.size() - matchedCFOS.size()

    print '  GFP+ single:   ' + nGFPonly
    print '  cFOS+ single:  ' + nCFOSonly
    print '  Double pos.:   ' + nDouble

    hierarchy.fireHierarchyChangedEvent(this)
    entry.saveImageData(imageData)

    csvFile.append(prefix + ',' + nGFPonly + ',' + nCFOSonly + ',' + nDouble + ',' + gfpObjects.size() + ',' + cfosObjects.size() + '\n')
}

print '================================'
print 'Done! CSV: ' + csvPath
print '================================'