import coordinates
from PIL import Image
import cv2
import numpy as np
img = cv2.imread('Sample images/test_img_mask1.png')

thresh=100


img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_contours = cv2.imread('Sample images/test_img_mask2.png')

for c in contours:

    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img_contours, (x, y), (x+w,y+h), 255, 1)


cv2.imwrite('Sample images/contours.png',img_contours)