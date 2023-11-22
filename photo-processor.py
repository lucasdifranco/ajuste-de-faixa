import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class detlane():

    def __init__(self) -> None:

        self.image_path = "./image/teste.jpg"
        self.kernel = 5
        canny = self.canny_img(self.image_path)
        pass

    def canny_img(self,image_path) -> np.ndarray:
        '''
        Creates a canne-edge detection based on the original image.

        Parameters:
            image_path (path)
            kernel (int)
        Returns:
            canny (ndarray)

        '''
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        blur = cv.GaussianBlur(img,(self.kernel,self.kernel),0) # fix the blur of the image, you can change the kernel in __init__
        canny = cv.Canny(blur, 200, 300) # you can change threshold 1 and 2 to fit your images

        return canny
    


detlane()