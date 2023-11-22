import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class detlane():

    def __init__(self) -> None:

        self.image_path = "./image/teste.jpg"
        self.kernel = 5
        canny = self.canny_img(self.image_path)
        region_img = self.region(canny)
        cv.imshow("CANNY", region_img)
        cv.waitKey(0)
        pass

    def canny_img(self,img_path) -> np.ndarray:
        '''
        Creates a canne-edge detection based on the original image.

        Parameters:
            image_path (path)
            kernel (int)
        Returns:
            canny (ndarray)

        '''
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        blur = cv.GaussianBlur(img,(self.kernel,self.kernel),0) # fix the blur of the image, you can change the kernel in __init__
        canny = cv.Canny(blur, 200, 300) # you can change threshold 1 and 2 to fit your images

        return canny

    def region(self,img:np.ndarray) -> np.ndarray:
        '''
        Sets a region of interest in the image (removes noise around the lane).
        In this case im using a polygon so i can set the "height" of the the pavement sample, however you can use a triangle to fit your needs.

        Parameters:
            img (canny)
        Returns:
            mask
        '''

        height, width = img.shape

        img_pol = np.array([
                       [(0, height),
                        (width // 5, height // 3),
                        (3 * width // 4, height // 3),
                        (width, height)]
                       ])
        
        mask = np.zeros_like(img)

        mask = cv.fillPoly(mask,[img_pol],255)
        mask = cv.bitwise_and(img,mask)

        return mask

    
detlane()