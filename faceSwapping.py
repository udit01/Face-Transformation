import cv2
import numpy as np
import argparse
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve
from scipy import sparse
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
        # self.constructIntermediateImage()
        self.constructIntermediateImageBlending()
    
    def getPointsImage1(self, event, x, y, params, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.list1.append((y, x))    # because x is column and y is row
            cv2.circle(self.image1, (x, y), 3, (0, 255, 0), 3)
            cv2.imshow(self.w1, self.image1)

    def getPointsImage2(self, event, x, y, params, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.list2.append((y, x))
            cv2.circle(self.image2, (x, y), 3, (0, 0, 255), 3)
            cv2.imshow(self.w2, self.image2)
    
    def barycentric_coordinates(self, coordinates, triangle_numbers):
        bary_coord = []
        for i in range(len(triangle_numbers)):
            b = list(self.tri.transform[triangle_numbers[i],:2].dot(coordinates[i]-self.tri.transform[triangle_numbers[i],2]))
            b.append(1.0 - sum(b))
            bary_coord.append(b)
        return bary_coord

    def in_delta_omega(self,pixel):
        x,y = pixel
        m,n,_ = self.i2Shape
        if self.inside[x*n + y] < 0 and (self.inside[(x+1)*n + y] >= 0 or self.inside[x*n+y+1] >= 0 or self.inside[(x-1)*n+y] >= 0 or self.inside[x*n+y-1] >= 0):
            return True
    

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
        self.inside = p
        # Blending
        """
        if p[k] is >= 0 then k belongs to Omega
        |N_p| = 4
        q \in N_p AND Omega 
        """
        interior_points = {}
        count = 0
        for i in range(m):
            for j in range(n):
                if p[i*n + j] >= 0:
                    interior_points[(i,j)] = count
                    count += 1
        bary_coordinates = self.barycentric_coordinates(final_image_coordinates, p)
        simplices = self.tri.simplices
        int_image1 = np.zeros((m,n,3))
        top_pixel = np.zeros((m,n,3))
        right_pixel = np.zeros((m,n,3))
        down_pixel = np.zeros((m,n,3))
        left_pixel = np.zeros((m,n,3))
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
                top_pixel[i,j] = self.image1_unmod[(p1[0]-1,p1[1])]
                right_pixel[i,j] = self.image1_unmod[(p1[0],p1[1]+1)]
                down_pixel[i,j] = self.image1_unmod[(p1[0]+1,p1[1])]
                left_pixel[i,j] = self.image1_unmod[(p1[0],p1[1]-1)]
        # Now p has the triangle number for each of the pixels
        grad_x = cv2.Sobel(int_image1,cv2.CV_8U,1,0)  
        grad_y = cv2.Sobel(int_image1,cv2.CV_8U,0,1)
        row_indices = []
        column_indices = []
        data = []
        right_r = []
        right_g = []
        right_b = []
        for i in range(m):
            for j in range(n):
                triangle_number = p[i*n + j]
                if triangle_number < 0:
                    continue
                row_indices.append(interior_points[(i,j)])
                column_indices.append(interior_points[(i,j)])
                data.append(4)
                right = np.zeros(3)
                count = 0
                for neighbor in [(i+1,j),(i-1,j), (i,j+1), (i,j-1)]:
                    count += 1
                    if self.inside[neighbor[0]*n+neighbor[1]] >= 0:
                        row_indices.append(interior_points[(i,j)])
                        column_indices.append(interior_points[neighbor])
                        data.append(-1)
                    elif self.in_delta_omega(neighbor):
                        right += self.image2_unmod[neighbor]
                        if count == 1:
                            right += int_image1[i,j] - down_pixel[i,j]
                        if count == 2:
                            right += int_image1[i,j] - top_pixel[i,j]
                        if count == 3:
                            right += int_image1[i,j] - right_pixel[i,j]
                        if count == 4:
                            right += int_image1[i,j] - left_pixel[i,j]
                        continue
                    right += int_image1[i,j] - int_image1[neighbor]
                right_r.append(right[0])
                right_g.append(right[1])
                right_b.append(right[2])
        print("Start")
        A = sparse.csr_matrix((data,(row_indices, column_indices)),shape=(len(interior_points),len(interior_points)))
        b_r = np.array(right_r)
        b_g = np.array(right_g)
        b_b = np.array(right_b)
        f_r = spsolve(A,b_r)
        print("Done1")
        f_g = spsolve(A,b_g)
        print("Done2")
        f_b = spsolve(A,b_b)
        print("Done3")
        print(len(interior_points))
        # cv2.imwrite('final.png',final_image)
        cv2.imwrite('before_change_swap.png', int_image1)
        for i in range(m):
            for j in range(n):
                if self.inside[i*n + j] < 0:
                    continue
                int_image1[i,j][0] = f_r[interior_points[(i,j)]]
                int_image1[i,j][1] = f_g[interior_points[(i,j)]]
                int_image1[i,j][2] = f_b[interior_points[(i,j)]]
        cv2.imwrite('final_swap.png', int_image1)


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Morph some images')
    parser.add_argument('-i1', dest='i1path', help='The image 1 path', required = True, type=str)
    parser.add_argument('-i2', dest='i2path', help='The image 2 path', required = True, type=str)
    # parser.add_argument('-k', dest='k', help='Morphing through k frames', required = True, type=int, default=1)
    args = parser.parse_args()

    swapper = Swapper()
    swapper.loadImages(args.i1path, args.i2path)
    swapper.loadWindows()