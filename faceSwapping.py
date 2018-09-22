import cv2
import numpy as np
import argparse
from scipy.spatial import Delaunay
import itertools

class Swapper:

    def __init__(self):
        self.init_lists()
    
    def init_lists(self):
        self.list1 = []
        self.list2 = []
    
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
        cv2.imshow(self.w1, self.image1)
        cv2.imshow(self.w2, self.image2)
        cv2.setMouseCallback(self.w1, self.getPointsImage1)
        cv2.setMouseCallback(self.w2, self.getPointsImage2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.constructIntermediateImage()
    
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
    
    def barycentric_coordinates(self, coordinates, triangle_numbers):
        bary_coord = []
        for i in range(len(triangle_numbers)):
            b = list(self.tri.transform[triangle_numbers[i],:2].dot(coordinates[i]-self.tri.transform[triangle_numbers[i],2]))
            b.append(1.0 - sum(b))
            bary_coord.append(b)
        return bary_coord

    def constructIntermediateImage(self):
        myadd = lambda xs,ys: tuple(int(x + y) for x, y in zip(xs, ys))
        self.inter_image = self.list2
        print(self.inter_image)
        self.tri = Delaunay(self.inter_image)
        # print(tri.simplices)
        # tri.simplices is a 2d-numpy array kx3 array with indices of points of each simplex in each row
        m,n,_ = self.i2Shape
        print((m,n))
        final_image_coordinates = list(itertools.product(range(m),range(n)))
        p = self.tri.find_simplex(final_image_coordinates)
        # interior_points = {}
        bary_coordinates = self.barycentric_coordinates(final_image_coordinates, p)
        simplices = self.tri.simplices
        final_image = np.zeros((m,n,3))
        for i in range(m):
            for j in range(n):
                triangle_number = p[i*n+j]
                if triangle_number < 0:
                    final_image[i,j] = self.image2_unmod[i,j]
                    continue
                corner_points = simplices[triangle_number]
                barycentric = bary_coordinates[i*n+j]
                p1 = myadd(myadd(tuple(x*barycentric[0] for x in self.list1[corner_points[0]]),tuple(x*barycentric[1] for x in self.list1[corner_points[1]])),tuple(x*barycentric[2] for x in self.list1[corner_points[2]]))
                final_image[i,j] = 0.7*self.image1_unmod[p1]+0.3*self.image2_unmod[i,j]
        # Now p has the triangle number for each of the pixels
        cv2.imwrite('final.png',final_image)

    def constructIntermediateImageBlending(self):
        myadd = lambda xs,ys: tuple(int(x + y) for x, y in zip(xs, ys))
        self.inter_image = self.list2
        print(self.inter_image)
        self.tri = Delaunay(self.inter_image)
        # print(tri.simplices)
        # tri.simplices is a 2d-numpy array kx3 array with indices of points of each simplex in each row
        m,n,_ = self.i2Shape
        print((m,n))
        final_image_coordinates = list(itertools.product(range(m),range(n)))
        p = self.tri.find_simplex(final_image_coordinates)
        interior_points = {}
        count = 0
        for i in range(m):
            for j in range(n):
                if p[i*n + j] >= 0:
                    interior_points[(i,j)] = count
                    count += 1
        A = []
        b = []
        bary_coordinates = self.barycentric_coordinates(final_image_coordinates, p)
        simplices = self.tri.simplices
        int_image1 = np.zeros((m,n,3))
        for i in range(m):
            for j in range(n):
                triangle_number = p[i*n+j]
                if triangle_number < 0:
                    int_image1[i,j] = self.image2_unmod[i,j]
                    continue
                corner_points = simplices[triangle_number]
                barycentric = bary_coordinates[i*n+j]
                p1 = myadd(myadd(tuple(x*barycentric[0] for x in self.list1[corner_points[0]]),tuple(x*barycentric[1] for x in self.list1[corner_points[1]])),tuple(x*barycentric[2] for x in self.list1[corner_points[2]]))
                int_image1[i,j] = self.image1_unmod[p1]
        # Now p has the triangle number for each of the pixels
        border = lambda tup : True if (p[(tup[0]+1)*n + tup[1]] < 0 or p[(tup[0]-1)*n+tup[1]] < 0 or p[tup[0]*n + tup[1]+1] < 0 or p[tup[0]*n + tup[1]-1] < 0) else False
        for i in range(m):
            for j in range(n):
                if p[i*n + j] < 0:
                    l = [0*len(interior_points)]
                    l[interior_points[(i,j)]] = 4

        # cv2.imwrite('final.png',final_image)




if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Morph some images')
    parser.add_argument('-i1', dest='i1path', help='The image 1 path', required = True, type=str)
    parser.add_argument('-i2', dest='i2path', help='The image 2 path', required = True, type=str)
    # parser.add_argument('-k', dest='k', help='Morphing through k frames', required = True, type=int, default=1)
    args = parser.parse_args()

    swapper = Swapper()
    swapper.loadImages(args.i1path, args.i2path)
    swapper.loadWindows()