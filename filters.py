import cv2
from scipy.spatial import Delaunay
import numpy as np
import argparse
import itertools
class Filter:

    def __init__(self):
        self.init_lists()

    def loadImages(self, path1):
        self.image1 = cv2.imread(path1)
        self.image1_unmod = cv2.imread(path1)
        self.i1Shape = self.image1.shape
    
    def loadWindows(self):
        self.w1 = 'Mark Tranformation Points for filter number'

        cv2.namedWindow(self.w1, cv2.WINDOW_NORMAL)

        cv2.imshow(self.w1, self.image1)
        cv2.setMouseCallback(self.w1, self.getPointsImage1)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.extendLists()
        self.printLists()

        self.constructFilteredImage(0)
    
    def constructFilteredImage(self, idx)
        
        # Transform self.filters(idx) with cv2.effine tranform,
        B = np.array(self.image1, dtype=np.float) 

        rows,cols,ch = self.image1.shape

        # get from points array and marked things
        # p_1 = self.points[3*idx + 0]
        # p_2 = self.points[3*idx + 1]
        # p_3 = self.points[3*idx + 2]

        # Will throw some error if < 3 points
        # target_p_1 = self.list1[0] 
        # target_p_2 = self.list1[1] 
        # target_p_3 = self.list1[2] 
        
        # pts1 = np.float32([p_1,p_2,p_3])
        # pts2 = np.float32([ target_p_1, target_p_2 , target_p_3 ])

        pts1 = np.float32(self.points[ 3*idx : 3*(idx+1) ])
        pts2 = np.float32(self.list1[ 0 : 3 ])

        M = cv2.getAffineTransform(pts1,pts2)

        # will extract view of (cols, rows)
        A = cv2.warpAffine(self.filters[idx], M, (cols,rows))

        alpha = self.alphas[idx]

        # background = cv2.imread('road.jpg')
        # overlay = cv2.imread('traffic sign.png')
        # rows,cols,channels = overlay.shape
        # overlay=cv2.addWeighted(background[250:250+rows, 0:0+cols],0.5,overlay,0.5,0)
        # background[250:250+rows, 0:0+cols ] = overlay

        # A is foreground, B is background
        output = (A * alpha) + (B * (1-alpha))
        # output = (A * alpha) + (B)

        cv2.imwrite('filter_'+idx+'.png' , output)

        cv2.imshow('OUTPUT WITH FILTER '+ idx, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def init_lists(self):
        self.list1 = []
    
    def loadFilters(self, dir='./filters/'):
        self.filters = []
        self.filter_paths = []
        
        self.getPointsAndAplha()

        # with open(dir+'points.txt') as f:
        #     content = f.readlines()
        # # you may also want to remove whitespace characters like `\n` at the end of each line
        # content = [x.strip() for x in content]  

        for (dirpath, dirnames, filenames) in os.walk(dir):
            for filename in filenames:
                if filename.endswith('.png'): 
                    self.filter_paths.append(os.sep.join([dirpath, filename]))

        for image_path in self.filter_paths:
            image = cv2.imread(image_path)
            self.filters.append(image)

        # Filters, points and alpha's loaded
        

    def getPointsAndAplha(self):
        
        #These are the 3 hot points for effine tranform for each filters
        self.points = []
        self.alphas = []

        # 3 points & 1 alpha for 0.png
        self.points.append((0,5)) 
        self.points.append((0,5)) 
        self.points.append((0,5)) 

        self.alphas.append(0.9)
        # have a map or number for opacity

        # 3 points & 1 alpha for 0.png


    def getPointsImage1(self, event, x, y, params, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.list1.append((y,x))    # because x is column and y is row
            cv2.circle(self.image1, (x,y), 3, (0,255,0), 3)
            cv2.imshow(self.w1, self.image1)

    def printLists(self):
        print("------------- Current list--------------")
        print(self.list1)
        print("------------- Points  list--------------")
        print(self.points)
        print("------------- Alphas  list--------------")
        print(self.alphas)
        print("------------- ImgPath list--------------")
        print(self.filter_paths)


    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Morph some images')
    parser.add_argument('-i', dest='ipath', help='The image path', required = True, type=str)
    parser.add_argument('-fp', dest='fp', help='The filter paths', default = './filters/', required = True, type=str)
    parser.add_argument('-n', dest='num', help='The filter number', default = 0, required = True, type=int)
    args = parser.parse_args()

    fil = Filter(args.num)
    fil.loadImages(args.ipath)
    fil.loadFilters(args.fp)
    fil.loadWindows()

    
    