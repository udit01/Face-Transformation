import os
import cv2
from scipy.spatial import Delaunay
import numpy as np
import argparse
import itertools

def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # Black Background Image
    black_background_image = np.zeros_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on black Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    black = black_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + black
    return final_image.astype(np.uint8)    


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

        self.printLists()

        # self.constructFilteredImage(0)
    
    def constructFilteredImage(self, idx):
        
        # Transform self.filters(idx) with cv2.effine tranform,
        B = np.array(self.image1_unmod, dtype=np.float) 
        # B /= 255.0 
        rows,cols,ch = self.image1_unmod.shape

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
        pts2 = np.float32(self.list1[ 0 : 3 ]) # target points

        M = cv2.getAffineTransform(pts1,pts2)

        # will extract view of (cols, rows)
        A = cv2.warpAffine(self.filters[idx], M, (cols,rows))
        # A /= 255.0
        self.writeAndShow(A, 'ToungeWarped.png')

        alpha = self.alphas[idx]

        # background = cv2.imread('road.jpg')
        # overlay = cv2.imread('traffic sign.png')
        # rows,cols,channels = overlay.shape
        # overlay=cv2.addWeighted(background[250:250+rows, 0:0+cols],0.5,overlay,0.5,0)
        # background[250:250+rows, 0:0+cols ] = overlay

        # A is foreground, B is background
        # smartly adding with alpha channel (modification in read_transparent func) ?
        # print(A[0][0])
        output = self.overlay(A, alpha, B)
        # output = (A * alpha) + (B * (1-alpha))
        # output = (A * alpha) + (B)
        
        self.writeAndShow(output, 'filter_'+str(idx)+'.png')

    def overlay(self, A, alpha, B):
        output = np.zeros_like(B)
        row, col, ch = output.shape
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        
        for r in range(row):
            for c in range(col):
                if ((A[r][c] == np.array([0, 0, 0])).all()):
                    output[r][c] = B[r][c]
                else:
                    o = np.add(  np.multiply(alpha,A[r][c]), np.multiply((1-alpha),B[r][c])  )
                    output[r][c] = o.astype(np.int)
        
        return output

    def writeAndShow(self, img, name='a.png'):

        cv2.imwrite(name , img)

        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def init_lists(self):
        self.list1 = []
    
    def loadFilters(self, dir='filters/'):
        self.filters = []
        self.filter_paths = []
        
        self.getPointsAndAplha()

        # with open(dir+'points.txt') as f:
        #     content = f.readlines()
        # # you may also want to remove whitespace characters like `\n` at the end of each line
        # content = [x.strip() for x in content]  

        # for (dirpath, dirnames, filenames) in os.walk(dir):
        #     for filename in filenames:
        #         if filename.endswith('.png'): 
        #             self.filter_paths.append(os.sep.join([dirpath, filename]))

        for i in range(5):
            self.filter_paths.append(os.path.join(dir, str(i)+'.png' ))

        for image_path in self.filter_paths:
            image = read_transparent_png(image_path)
            self.filters.append(image)

        # Filters, points and alpha's loaded
        

    def getPointsAndAplha(self):
        
        #These are the 3 hot points for effine tranform for each filters
        self.points = []
        self.alphas = []

        # Got from https://www.mobilefish.com/services/record_mouse_coordinates/record_mouse_coordinates.php
        # inverted x and y to get row and col

        # 3 points & 1 alpha for 0.png "Specs"
        self.points.append([40, 60]) # left eye center
        self.points.append([40, 185]) # right eye center 
        self.points.append([20, 120]) # center of spects

        self.alphas.append(0.5)
        # have a map or number for opacity

        # 3 points & 1 alpha for 1.png "Tongue"
        # self.points.append([35, 95]) # Top center
        # self.points.append([164, 3]) # Leftmost 
        # self.points.append([170, 194]) # RightMost

        # self.alphas.append(0.5)
        # reversed x and y
        self.points.append([95, 35]) # Top center
        self.points.append([3, 164]) # Leftmost 
        self.points.append([194, 160]) # RightMost

        self.alphas.append(0.95)



        # 3 points & 1 alpha for 2.png "Mustache" 
        self.points.append([29, 328]) # Leftmost
        self.points.append([24, 93]) # Middle 
        self.points.append([28, 184]) # RightMost

        self.alphas.append(0.5)


        # 3 points & 1 alpha for 3.png # "Hat"
        self.points.append([9, 117]) # Top Center
        self.points.append([94, 2]) # Leftmost
        self.points.append([94, 233]) # RightMost

        self.alphas.append(0.5)


        # 3 points & 1 alpha for 4.png #"Dog-nose"
        self.points.append([74, 2]) # Leftmost
        self.points.append([39, 91]) # Middle 
        self.points.append([76, 185]) # RightMost

        self.alphas.append(0.5)


    def getPointsImage1(self, event, x, y, params, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.list1.append([x,y])    # because x is column and y is row
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

    fil = Filter()
    fil.loadImages(args.ipath)
    fil.loadFilters(args.fp)
    fil.loadWindows()
    fil.constructFilteredImage(args.num)

    
###
# import cv2
# img = cv2.imread('filters/1.png', cv2.IMREAD_UNCHANGED)
# img2 = cv2.imread('filters/1.png')

# cv2.imshow("1", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow("2", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()