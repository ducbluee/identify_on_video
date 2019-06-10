import cv2
import numpy as np 
import time
import imutils
from plate_detection import plate
from matplotlib import pyplot as plt
from classCNN import NeuralNetwork
from skimage.filters import threshold_local
from skimage import measure

class character():
    def __init__(self, crop_contour):
        self.crop_contour = crop_contour

    def sort_contours(self,new_contours):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in new_contours]
        (new_contours, boundingBoxes) = zip(*sorted(zip(new_contours, boundingBoxes),
            key=lambda b:b[1][i], reverse=False))
        return new_contours

    def find_character(self, crop_contour, fixed_width):
        V = cv2.split(cv2.cvtColor(self.crop_contour, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 29, offset=15, method='gaussian')
        thresh = (V > T).astype('uint8') * 255
        thresh =  cv2.bitwise_not(thresh)
        self.crop_contour = imutils.resize(self.crop_contour, width=fixed_width)
        thresh = imutils.resize(thresh, width=fixed_width)
        self.bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return thresh, self.bgr_thresh
    
    def segment_character(self, thresh, crop_contour):
        labels = measure.label(thresh, neighbors=8, background=0)
        charCandidates = np.zeros(thresh.shape, dtype='uint8')
        characters = []
        for label in np.unique(labels):
            if label == 0:
                continue
            labelMask = np.zeros(thresh.shape, dtype='uint8')
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(self.crop_contour.shape[0])
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.5 and heightRatio < 0.95
                if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)
        #cv2.imshow('char',charCandidates)
        return charCandidates

    def length(self, charCandidates):
        length = []
        _, list_contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in list_contours:
            length.append(c)
        lenL = len(length)
        #print(lenL)
        return lenL

    # def clean(self,charCandidates,bgr_thresh):
    #     _, new_contours, hier = cv2.findContours(charCandidates, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     new_contours = sorted(new_contours, key=cv2.contourArea, reverse=True)[:10]
    #     new_contours = self.sort_contours(new_contours)
    #     print(1)

    def read(self, charCandidates, bgr_thresh):
        # myNetwork = NeuralNetwork(modelFile="model/ducdn_ver3.pb",labelFile="model/ducdn_ver3.txt")
        # List = []
        # plate = ""
        addPixel = 4
        _, new_contours, hier = cv2.findContours(charCandidates, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = sorted(new_contours, key=cv2.contourArea, reverse=True)[:10]
        if new_contours:
            new_contours = self.sort_contours(new_contours)
            characters = []
            addPixel = 4
            for c in new_contours:
                (x,y,w,h) = cv2.boundingRect(c)
                if y > addPixel:
                    y = y - addPixel
                else:
                    y = 0
                if x > addPixel:
                    x = x - addPixel
                else:
                    x = 0
                temp = bgr_thresh[y:y+h+(addPixel*2), x:x+w+(addPixel*2)]
                # cv2.imshow('temp', temp)
                # cv2.waitKey(0)
                # cv2.imwrite("d/d"+ str(time.time())+'.jpg',temp)
                # tensor = myNetwork.read_tensor_from_image(temp,224)
                # label = myNetwork.label_image(tensor)
                characters.append(temp)
                cv2.imwrite("file_1/f"+ str(time.time())+'.jpg',temp)
            return characters
        else:
            return None
