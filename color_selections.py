import numpy as np
import cv2 as cv

class color_selection():

    def __init__(self) -> None:

        pass

    def RGB_selection(img) -> np.ndarray:
        '''
        Masks the image to search for yellow and white lines.

        Parameters:
            img (np.ndarray)

        Returns:
            masked_image

        '''
        
        lower_threshold = np.uint8([200, 200, 200])
        upper_threshold = np.uint8([255, 255, 255])
        white_mask = cv.inRange(img, lower_threshold, upper_threshold)
        
        #Yellow color mask
        # lower_threshold = np.uint8([125, 140,   0])
        # upper_threshold = np.uint8([255, 255, 255])

        lower_threshold = np.uint8([115, 140,  150])
        upper_threshold = np.uint8([255, 255, 255])
        yellow_mask = cv.inRange(img, lower_threshold, upper_threshold)
        
        #Combine white and yellow masks
        mask = cv.bitwise_or(white_mask, yellow_mask)
        rgb_image = cv.bitwise_and(img, img, mask = mask)

        return rgb_image
    
    def HSV_selection(img) -> np.ndarray:
        '''
        Masks the image to search for yellow and white lines.

        Parameters:
            img (np.ndarray)

        Returns:
            masked_image

        '''
        
        img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        lower_threshold = np.uint8([0, 0, 210])
        upper_threshold = np.uint8([255, 30, 255])
        white_mask = cv.inRange(img, lower_threshold, upper_threshold)
        
        #Yellow color mask
        lower_threshold = np.uint8([20, 40, 60])
        upper_threshold = np.uint8([30, 255, 255])
        yellow_mask = cv.inRange(img, lower_threshold, upper_threshold)
        
        #Combine white and yellow masks
        mask = cv.bitwise_or(white_mask, yellow_mask)
        hsv_image = cv.bitwise_and(img, img, mask = mask)

        return hsv_image 
    
    def HLS_selection(img) -> np.ndarray:
        '''
        Masks the image to search for yellow and white lines.

        Parameters:
            img (np.ndarray)

        Returns:
            masked_image (np.ndarray)

        '''
        
        img = cv.cvtColor(img, cv.COLOR_RGB2HLS)

        lower_threshold = np.uint8([0, 200, 0])
        upper_threshold = np.uint8([255, 255, 255])
        white_mask = cv.inRange(img, lower_threshold, upper_threshold)
        
        #Yellow color mask
        lower_threshold = np.uint8([30, 60, 169])
        upper_threshold = np.uint8([40, 255, 255])
        yellow_mask = cv.inRange(img, lower_threshold, upper_threshold)
        
        #Combine white and yellow masks
        mask = cv.bitwise_or(white_mask, yellow_mask)
        hsv_image = cv.bitwise_and(img, img, mask = mask)

        return hsv_image