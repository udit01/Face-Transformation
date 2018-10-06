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
    img = cv2.medianBlur(img,3)
    sobel = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize = 5)
    m,n = img.shape
    intermediate_image = np.ones((m,n), dtype = np.uint8)*255
    f = 11
    threshold = 70
    for i in range(1,m-1):
        for j in range(1,n-1):
            s = sobel[i,j]/255.0
            g = img[i,j]/255.0
            z = 0.4
            if s <= 0.5:
                z = 0.8
            w = (z*(1-s) + (1-z)*(1-g))*(f-8) + 8
            boost = high_boost(img, i, j, w)
            # intermediate_image[i,j] = boost
            if boost <= threshold:
                intermediate_image[i,j] = 0
    # intermediate_image = cv2.medianBlur(img, 3)
    intermediate_image = cv2.medianBlur(intermediate_image,3)
    cv2.imwrite('test.jpg',intermediate_image)
    return intermediate_image

def separation(image):
    img = image.copy()
    image = cv2.bitwise_not(image)
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    boxes = [] # list of the tuples 
    for i, contour in enumerate(contours):
     x, y, w, h = cv2.boundingRect(contour)
     boxes.append((x,y,w,h,contour[i]))
     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite('output3.png', img)
    return boxes

def distance(box1, box2):
    # Assuming non-overlapping boxes
    if overlap(box1, box2):
        return 0
    (x1, y1, h1, w1) = box1
    (x2, y2, h2, w2) = box2
    dist1 = 1000
    dits2 = 1000
    if x1 + w1 < x2 :
        dist1 = x2 - (x1 + w1)
    if x2 + w2 < x1 :
        dist1 = x1 - (x2 + w2)
    if y1 + h1 < y2 :
        dist2 = y2 - (y1 + h1)
    if y2 + h2 < y1:
        dist2 = y1 - (y2 + h2)
    return min(dist1, dist2)


def merge(boxes):
    boxes.sort(key= lambda x: x[2]*x[3])





image = cv2.imread('images/portrait2.jpg',cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)
prep_image = preprocess(image)
separation(prep_image)