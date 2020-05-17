import cv2
import numpy as np

class ColorRangeHSV:
    def __init__(self, lowerBound, upperBound):
        self.lowerBound = lowerBound
        self.upperBound = upperBound

class RangePickerHSV:
    def __init__(self, windowName):
        self.windowName = windowName

    def pickRange(self, imageBGR):
        x0, y0, x1, y1 = self.__selectObject(imageBGR)
        selectedAreaBGR = imageBGR[int(y0):int(y0 + y1), int(x0):int(x0 + x1)]
        selectedAreaHSV = cv2.cvtColor(selectedAreaBGR, cv2.COLOR_BGR2HSV)
        lowerBound = ([selectedAreaHSV[:,:,0].min(), selectedAreaHSV[:,:,1].min(), selectedAreaHSV[:,:,2].min()])
        upperBound = ([selectedAreaHSV[:,:,0].max(), selectedAreaHSV[:,:,1].max(), selectedAreaHSV[:,:,2].max()])
        return ColorRangeHSV(lowerBound, upperBound)

    def __selectObject(self, imageBGR):
        showCrosshair = False
        fromCenter = False
        return cv2.selectROI(self.windowName, imageBGR, fromCenter, showCrosshair)
