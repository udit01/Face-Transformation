import cv2
from scipy.spatial import Delaunay
import numpy as np
import argparse
import itertools
class Morpher:

    def __init__(self, output):
        self.init_lists()
        self.prefix = output
        self.k = 1

    def loadImages(self, path1, path2):
        self.image1 = cv2.imread(path1)
        self.image2 = cv2.imread(path2)
        self.image1_unmod = cv2.imread(path1)
        self.image2_unmod = cv2.imread(path2)
        self.i1Shape = self.image1.shape
        self.i2Shape = self.image2.shape
    
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
        self.extendLists()
        self.printLists()
        # self.constructIntermediateLists()
        for w in [0.33, 1.00, 3]:
            self.w = w
            self.constructIntermediateImage()
    
    def init_lists(self):
        self.list1 = []
        self.list2 = []
    
    def getPointsImage1(self, event, x, y, params, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.list1.append((y,x))    # because x is column and y is row
            cv2.circle(self.image1, (x,y), 3, (0,255,0), 3)
            cv2.imshow(self.w1, self.image1)

    def getPointsImage2(self, event, x, y, params, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.list2.append((y,x))
            cv2.circle(self.image2, (x,y), 3, (0,0,255), 3)
            cv2.imshow(self.w2, self.image2)

    def extendLists(self):
        # add the i1 and i2 corner coordinates
        
        # trim both of them to shorter length
        self.len = len(self.list2) if (len(self.list1) > len(self.list2)) else len(self.list1) 

        self.list1 = self.list1[:self.len]
        self.list2 = self.list2[:self.len]

        self.list1.append((0, 0))
        self.list1.append((0, self.i1Shape[1]))
        self.list1.append((self.i1Shape[0], 0))
        self.list1.append((self.i1Shape[0], self.i1Shape[1]))

        self.list2.append((0, 0))
        self.list2.append((0, self.i2Shape[1]))
        self.list2.append((self.i2Shape[0], 0))
        self.list2.append((self.i2Shape[0], self.i2Shape[1]))

        self.len = self.len + 4

    def constructIntermediateLists(self):
        k = self.k 
        self.inter_points = [[] for i in range(k)]
        self.inter_sizes  = []
        for i in range(k):
            left_weight = (float(k-i))/(float(k+1))
            right_weight = (float(i+1))/(float(k+1))
            
            # do down size in INT
            self.inter_sizes.append(((left_weight*(np.array(self.i1Shape).astype(float))) + (right_weight*(np.array(self.i2Shape).astype(float)))).astype(int))

            for j in range(self.len):
                self.inter_points[i].append((left_weight*((np.array(self.list1[j]).astype(float))) + right_weight*((np.array(self.list2[j]).astype(float)))).astype(int))  
            
    def constructIntermediateImages(self):
        self.inter_images = [np.zeros(s) for s in self.inter_sizes]
        self.triangulations = []
        for i in range(self.k):
            # print(len(self.inter_points[i]))
            # print(len(self.inter_points[i][0]))
            # print(len(self.inter_points[i][0].shape))
            tri = Delaunay(self.inter_points[i])
            # print(points[tri.simplices])
            self.triangulations.append(tri)

            # Use this link for Barycentric coordinates and effine transform
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html

    def barycentric_coordinates(self, coordinates, triangle_numbers):
        bary_coord = []
        for i in range(len(triangle_numbers)):
            b = list(self.tri.transform[triangle_numbers[i],:2].dot(coordinates[i]-self.tri.transform[triangle_numbers[i],2]))
            b.append(1.0 - sum(b))
            bary_coord.append(b)
        return bary_coord

    def constructIntermediateImage(self):
        w1 = 1.0/(1.0 + self.w)
        w2 = self.w/(1.0 + self.w)
        myadd = lambda xs,ys: tuple(int(x + y) for x, y in zip(xs, ys))
        self.inter_image = [myadd(tuple(w1*i for i in x) , tuple(w2*i for i in y)) for x,y in zip(self.list1, self.list2)]
        print(self.inter_image)
        self.tri = Delaunay(self.inter_image)
        # print(tri.simplices)
        # tri.simplices is a 2d-numpy array kx3 array with indices of points of each simplex in each row
        m,n = self.inter_image[-1]
        final_image_coordinates = list(itertools.product(range(m),range(n)))
        p = self.tri.find_simplex(final_image_coordinates)
        bary_coordinates = self.barycentric_coordinates(final_image_coordinates, p)
        simplices = self.tri.simplices
        final_image = np.zeros((m,n,3))
        for i in range(m):
            for j in range(n):
                triangle_number = p[i*n+j]
                corner_points = simplices[triangle_number]
                barycentric = bary_coordinates[i*n+j]
                p1 = myadd(myadd(tuple(x*barycentric[0] for x in self.list1[corner_points[0]]),tuple(x*barycentric[1] for x in self.list1[corner_points[1]])),tuple(x*barycentric[2] for x in self.list1[corner_points[2]]))
                p2 = myadd(myadd(tuple(x*barycentric[0] for x in self.list2[corner_points[0]]),tuple(x*barycentric[1] for x in self.list2[corner_points[1]])),tuple(x*barycentric[2] for x in self.list2[corner_points[2]]))
                final_image[i,j] = [int(x) for x in self.image1_unmod[p1]*w1 + self.image2_unmod[p2]*w2]
        # Now p has the triangle number for each of the pixels
        cv2.imwrite(self.prefix+'_'+str(self.w)+'.png',final_image)


    def showTriangulation(self):
        pass
        # import matplotlib.pyplot as plt
        # plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        # plt.plot(points[:,0], points[:,1], 'o')
        # plt.show()

    def printLists(self):
        print(self.list1)    
        print(self.list2)


    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Morph some images')
    parser.add_argument('-i1', dest='i1path', help='The image 1 path', required = True, type=str)
    parser.add_argument('-i2', dest='i2path', help='The image 2 path', required = True, type=str)
    # parser.add_argument('-k', dest='k', help='Morphing through k frames', required = True, type=int, default=1)
    # parser.add_argument('-w', dest='w',help='Weight of image2 compared to image1', required=True,type=int,default=1)
    parser.add_argument('-o', dest='output', help='Output Prefix', required=True, type=str)
    args = parser.parse_args()

    morpher = Morpher(args.output)
    morpher.loadImages(args.i1path, args.i2path)
    morpher.loadWindows()

    
    