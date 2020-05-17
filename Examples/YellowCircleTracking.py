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

def generateMask(imageHSV, colorRange):
    lower = np.array(colorRange[0])
    upper = np.array(colorRange[1])
    return cv2.inRange(imageHSV, lower, upper)

def removeDots(maskedImage):
    kernel = np.ones((20, 20), np.uint8)
    maskedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_OPEN, kernel)
    maskedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_CLOSE, kernel)
    return maskedImage

def getBoundingRect(imageBGR, binaryImage):
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imageBGR, contours,-1, (0,255,0), 3)
    if len(contours) > 0:
        biggestContour = sorted(list(contours), key = lambda c: cv2.contourArea(c))[-1]
        epsilon = 0.02*cv2.arcLength(biggestContour, True)
        approx = cv2.approxPolyDP(biggestContour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        return x, y, x+w, y+h
    return -1, -1, -1, -1

def detectObjects(imageBGR, colorRangesHSV):
    foundObjects = []
    blurredBGR = cv2.GaussianBlur(imageBGR, (11, 11), 0)
    imageHSV = cv2.cvtColor(blurredBGR, cv2.COLOR_BGR2HSV)
    for colorRange in colorRangesHSV:
        binaryImage = removeDots(generateMask(imageHSV, colorRange))
        x0, y0, x1, y1 = getBoundingRect(imageBGR, binaryImage)
        if all(map(lambda c: c != -1, (x0, y0, x1, y1))):
            foundObjects.append(((x0, y0), (x1, y1)))
    return foundObjects

def extractFeatures(boundingRectCoords, frameShape):
    rectCenterX = int((boundingRectCoords[0][0] + boundingRectCoords[1][0])/2)
    rectCenterY = int((boundingRectCoords[0][1] + boundingRectCoords[1][1])/2)
    frameCenterX = int(frameShape[1] / 2)
    frameCenterY = int(frameShape[0] / 2)
    deltaX = frameCenterX - rectCenterX
    deltaY = frameCenterY - rectCenterY
    return (rectCenterX, rectCenterY), (frameCenterX, frameCenterY), (deltaX, deltaY)

def drawObjects(imageBGR, features):
    cv2.circle(imageBGR, features[0], 10, (255, 255, 255), -1, cv2.FILLED)
    cv2.circle(imageBGR, features[1], 10, (255, 255, 255), -1, cv2.FILLED)
    cv2.rectangle(imageBGR, features[0], features[1], (255, 255, 255), 3)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontThickness = 1
    textIndent = int(10)
    
    drawnText = "delta Y: {}".format(features[2][1])
    textCoord = (textIndent, int(imageBGR.shape[0] - textIndent))
    cv2.putText(imageBGR, drawnText, textCoord, fontFace, fontScale, (255, 255, 255), fontThickness)

    drawnText = "delta X: {}".format(features[2][0])
    textBoundingRect, _ = cv2.getTextSize(drawnText, fontFace, fontScale, fontThickness)
    textCoord = (textIndent, int(imageBGR.shape[0] - 2*textIndent - textBoundingRect[1]))
    cv2.putText(imageBGR, drawnText, textCoord, fontFace, fontScale, (255, 255, 255), fontThickness)

def waitKey():
    return cv2.waitKey(5) & 0xFF

def showImage(imageBGR):
    cv2.imshow(windowName, imageBGR)
    videoWriter.write(imageBGR)

def findObjects(colorRangesHSV):
    videoCapture = createVideoCapture()
    while True:
        succeeded, capturedImage = videoCapture.read()
        if not succeeded:
            continue
        boundingRects = detectObjects(capturedImage, colorRangesHSV)
        for r in boundingRects:
            features = extractFeatures(r, capturedImage.shape)
            drawObjects(capturedImage, features)
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
