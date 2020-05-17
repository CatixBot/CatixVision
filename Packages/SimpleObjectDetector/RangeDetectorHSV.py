import cv2
import numpy as np

from .RangePickerHSV import ColorRangeHSV

class ExtractedFeatures:
    def __init__(self, rangeRect, rangeContour, frameShape):
        rectCenterX = int((rangeRect[0][0] + rangeRect[1][0])/2)
        rectCenterY = int((rangeRect[0][1] + rangeRect[1][1])/2)
        frameCenterX = int(frameShape[1] / 2)
        frameCenterY = int(frameShape[0] / 2)
        axialDeltaX = frameCenterX - rectCenterX
        axialDeltaY = frameCenterY - rectCenterY
        self.detectedCenterPoint = (rectCenterX, rectCenterY)
        self.detectedContour = rangeContour
        self.frameCenterPoint = (frameCenterX, frameCenterY)
        self.axialDelta = (axialDeltaX, axialDeltaY)

class RangeDetectorHSV:
    def __init__(self, colorRangesHSV):
        self.__colorRangesHSV = colorRangesHSV

    def detectRange(self, imageBGR):
        detectedFeatures = []
        blurredBGR = self.__applyBlur(imageBGR)
        imageHSV = cv2.cvtColor(blurredBGR, cv2.COLOR_BGR2HSV)
        for colorRange in self.__colorRangesHSV:
            binaryImage = self.__removeDots(self.__calcRangeMask(imageHSV, colorRange))
            rangeRect, rangeContour = self.__getBoundingRect(binaryImage)
            if all(map(lambda c: c != -1, (rangeRect[0][0], rangeRect[0][1], rangeRect[1][0], rangeRect[1][1]))):
                detectedFeatures.append(ExtractedFeatures(rangeRect, rangeContour, imageBGR.shape))
        return detectedFeatures

    def __applyBlur(self, imageBGR):
        kernelSize = (11, 11)
        return cv2.GaussianBlur(imageBGR, kernelSize, 0)

    def __calcRangeMask(self, imageHSV, colorRange):
        lower = np.array(colorRange[0])
        upper = np.array(colorRange[1])
        return cv2.inRange(imageHSV, lower, upper)

    def __removeDots(self, binaryImage):
        kernel = np.ones((20, 20), np.uint8)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
        return binaryImage

    def __getBoundingRect(self, binaryImage):
        contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            biggestContour = sorted(list(contours), key = lambda c: cv2.contourArea(c))[-1]
            epsilon = 0.02*cv2.arcLength(biggestContour, True)
            approx = cv2.approxPolyDP(biggestContour, epsilon, True)
            x0, y0, w, h = cv2.boundingRect(approx)
            return ((x0, y0), (x0+w, y0+h)), biggestContour
        return ((-1, -1), (-1, -1)), []
