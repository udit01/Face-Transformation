import os
import cv2

import numpy as np
import argparse
import itertools
from automatic import *
class Landmark:

    def __init__(self):
        self.init_lists()

    def loadImages(self, path1):
        self.image1 = cv2.imread(path1)
        self.image1_unmod = cv2.imread(path1)
        self.i1Shape = self.image1.shape
        self.ipath = path1
    
    def loadWindows(self):
        self.w1 = 'Features automatically detected'

        cv2.namedWindow(self.w1, cv2.WINDOW_NORMAL)
        
        cv2.imshow(self.w1, self.image1)

        for feature in self.list1:
            cv2.circle(self.image1, feature, 3, (0, 255, 0), 3)
        
        cv2.imshow(self.w1, self.image1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def writeAndShow(self, img, name='a.png'):

        cv2.imwrite(name , img)

        # cv2.imshow(name, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def init_lists(self):
        self.list1 = []
    
    def showPointsImage1(self, event, x, y, params, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.list1.append([x,y])    # because x is column and y is row
            cv2.circle(self.image1, (x,y), 3, (0,255,0), 3)
            cv2.imshow(self.w1, self.image1)
            
    def getEyes(self):
        possible_eyes = getFeatures(self.ipath)

        for eyes in possible_eyes:
            self.list1.append((int(eyes.mean_x), int(eyes.mean_y)))

    def printLists(self):
        print("------------- Current list--------------")
        print(self.list1)

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Detect Facial Landmarks')
    parser.add_argument('-i', dest='ipath', help='The image path', required = True, type=str)
    args = parser.parse_args()

    la = Landmark()
    la.loadImages(args.ipath)
    la.getEyes()
    la.loadWindows()
