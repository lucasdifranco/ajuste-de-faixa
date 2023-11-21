import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
image_path = "./image/teste.jpg"
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
kernel = 5
blur = cv.GaussianBlur(img,(kernel,kernel),0)
canny = cv.Canny(blur, 200, 300)
cv.imshow("image",canny)
cv.waitKey(0)