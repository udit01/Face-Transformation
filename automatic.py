import cv2
import numpy as np

def high_boost(image, i, j, w):
    return_value = 0
    for x in range(i-1,i+1):
        for y in range(j-1,j+1):
            return_value -= image[x,y]
    return_value += (w+1)*image[i,j]
    return (return_value/9.0)


def preprocess(img):
    sobel = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize = 5)
    m,n = img.shape
    intermediate_image = np.ones((m,n))*255
    f = 11
    threshold = -200
    for i in range(1,m-1):
        for j in range(1,n-1):
            s = sobel[i,j]/255.0
            g = img[i,j]/255.0
            z = 0.4
            if s <= 0.5:
                z = 0.8
            w = (z*(1-s) + (1-z)*(1-g))*(f-8) + 8
            boost = high_boost(sobel, i, j, w)
            # intermediate_image[i,j] = boost
            if boost > threshold:
                intermediate_image[i,j] = 255
            else:
                intermediate_image[i,j] = 0
    cv2.imwrite('test.jpg',intermediate_image)


image = cv2.imread('images/portrait2.jpg',cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)
preprocess(image)

    
