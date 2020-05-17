import cv2
import numpy as np

import SimpleObjectTracker as SOT

windowName = "Pick&Find"
captureKey = ord('r')
exitKey = ord('q')

videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (640, 480))

def createVideoCapture():
    frameWidth = 480
    frameHeight = 640
    videoCapture = cv2.VideoCapture(0)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    return videoCapture

def drawTrackingRect(imageBGR, features):
    contourColor = (0,255,0)
    rectColor = (255, 255, 255)
    cv2.drawContours(imageBGR, [features.detectedContour], -1, contourColor, 3)
    cv2.circle(imageBGR, features.detectedCenterPoint, 10, rectColor, -1, cv2.FILLED)
    cv2.circle(imageBGR, features.frameCenterPoint, 10, rectColor, -1, cv2.FILLED)
    cv2.rectangle(imageBGR, features.detectedCenterPoint, features.frameCenterPoint, rectColor, 3)

def drawAnnotations(imageBGR, features):
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontThickness = 1
    fontColor = (255, 255, 255)
    textIndent = int(10) 
    drawnText = "delta Y: {}".format(features.axialDelta[1])
    textCoord = (textIndent, int(imageBGR.shape[0] - textIndent))
    cv2.putText(imageBGR, drawnText, textCoord, fontFace, fontScale, fontColor, fontThickness)
    drawnText = "delta X: {}".format(features.axialDelta[0])
    textBoundingRect, _ = cv2.getTextSize(drawnText, fontFace, fontScale, fontThickness)
    textCoord = (textIndent, int(imageBGR.shape[0] - 2*textIndent - textBoundingRect[1]))
    cv2.putText(imageBGR, drawnText, textCoord, fontFace, fontScale, fontColor, fontThickness)

def drawDetected(imageBGR, features):
    drawTrackingRect(imageBGR, features)
    drawAnnotations(imageBGR, features)

def waitKey():
    return cv2.waitKey(5) & 0xFF

def showImage(imageBGR):
    cv2.imshow(windowName, imageBGR)
    videoWriter.write(imageBGR)

def findObjects(colorRangesHSV):
    videoCapture = createVideoCapture()
    rangeTracker = SOT.RangeTrackerHSV(colorRangesHSV)
    while True:
        succeeded, capturedImage = videoCapture.read()
        if not succeeded:
            continue
        detectedFeatures = rangeTracker.detectRange(capturedImage)
        #boundingRects = detectObjects(capturedImage, colorRangesHSV)
        #for r in boundingRects:
        for features in detectedFeatures:
            drawDetected(capturedImage, features)
        showImage(capturedImage)
        keyPressed = waitKey()
        if keyPressed == exitKey:
            break
    videoCapture.release()

def pickObjects():
    colorRangesHSV = []
    videoCapture = createVideoCapture()
    rangePicker = SOT.RangePickerHSV(windowName)
    while True:
        succeeded, capturedImage = videoCapture.read() 
        if not succeeded:
            continue
        keyPressed = waitKey()
        if keyPressed == captureKey:
            colorRange = rangePicker.pickRange(capturedImage)
            colorRangesHSV.append((colorRange.lowerBound, colorRange.upperBound))
        else:
            showImage(capturedImage)
        if keyPressed == exitKey:
            break
    videoCapture.release()
    return colorRangesHSV

if __name__ == "__main__":
    findObjects(pickObjects())

videoWriter.release()
