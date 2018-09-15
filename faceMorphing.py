import cv2
import scipy
import numpy as np

class Morpher:
    def loadImages(self, path1, path2):
        self.image1 = cv2.imread(path1)
        self.image2 = cv2.imread(path2)
    
    def loadWindows(self):
        self.w1 = 'Get Image1 Points'
        self.w2 = 'Get Image2 Points'

        cv2.namedWindow(self.w1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.w2, cv2.WINDOW_NORMAL)

        cv2.imshow(self.w1, self.image1)
        cv2.imshow(self.w2, self.image2)
        cv2.setMouseCallback(self.w1, self.getPointsImage1)
        cv2.setMouseCallback(self.w2, self.getPointsImage2)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def init_lists(self):
        self.list1 = []
        self.list2 = []

    
    def getPointsImage1(self, event, x, y, params):
        if event == cv2.EVENT_LBUTTONCLICK:
            self.list1.append((y,x))    # because x is column and y is row

    def getPointsImage2(self, event, x, y, params):
        if event == cv2.EVENT_LBUTTONCLICK:
            self.list2.append((y,x))    
    
if __name__ == __main__ :
    morpher = Morph()

    