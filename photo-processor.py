import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class detlane():

    def __init__(self) -> None:

        self.image_path = "./image/teste.jpg"
        self.kernel = 5
        self.detlane()

        pass

    def detlane(self):
        '''
        run all the commands:
        first reads the image
        '''

        src_img = cv.imread(self.image_path)

        canny = self.canny_img(src_img) # changes the image to grayscale and uses canny to get the edges in the picture

        region_img = self.region(canny)

        lines = self.HoghLines(region_img) # get all lines possible
        right, left = self.average_lines(lines) # average of all lines

        r_points = self.make_points(src_img, right) # points for the right line lane
        l_points = self.make_points(src_img, left)  # points for the left line lane

        dst_img = self.img_lines(src_img,r_points,l_points)
        dst_img = cv.resize(dst_img,(1280 // 2, 720 // 2))

        cv.imshow('',dst_img)
        cv.waitKey(0)


    def canny_img(self, img) -> np.ndarray:
        '''
        Creates a canne-edge detection based on the original image.

        Parameters:
            image_path (path)
            kernel (int)
        Returns:
            canny (ndarray)

        '''
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        blur = cv.GaussianBlur(img, (self.kernel,self.kernel), 0) # fix the blur of the image, you can change the kernel in __init__
        canny = cv.Canny(blur, 200, 300) # you can change threshold 1 and 2 to fit your images

        return canny

    def region(self, img:np.ndarray) -> np.ndarray:
        '''
        Sets a region of interest in the image (removes noise around the lane).
        In this case im using a polygon so i can set the "height" of the the pavement sample, however you can use a triangle to fit all .

        Parameters:
            img (canny)
        Returns:
            mask
        '''

        height, width = img.shape

        img_pol = np.array([
                       [(-150, height),
                        (width // 5, height // 5),
                        (3 * width // 4, height // 5),
                        (width + 150, height)]
                       ])
        
        mask = np.zeros_like(img)

        mask = cv.fillPoly(mask, [img_pol], 255)
        mask = cv.bitwise_and(img,mask)

        return mask
    
    def HoghLines(self,img:np.ndarray) -> np.array:
        '''
        Transforms the clusters of white into actual lines in the image.
        This function gets all lines possible in the image, so after that you use a average function to get the desirable points.
        If you are not getting the lines that you want, please consder changing the threshold.

        Parameters:
            img (np.ndarray)
        Returns:
            lines (np.array)
        '''
        lines = cv.UMat(np.zeros((1, 1), dtype=np.int32))

        lines = cv.HoughLinesP(img, 
                                rho=2, 
                                theta=np.pi/180, 
                                threshold=100, 
                                lines=lines, 
                                minLineLength=40, 
                                maxLineGap=5)
        
        lines = np.array(lines.get())
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return lines
    
    def average_lines(self, lines:np.array) -> tuple[np.array,np.array]:
        '''
        Gets the average lines.

        Parameters:
            line (np.array)

        '''

        left = []
        right = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        
        return right_avg, left_avg

    def make_points(self,img:np.ndarray, average):
        '''
        Make points based o lines.
        '''
        slope, y_int = average 
        y1 = img.shape[0]
        y2 = int(y1 * (2/5))
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)

        return [(x1, y1), (x2, y2)]

    def img_lines(self,img:np.ndarray,r_points:list,l_points:list) -> np.ndarray:
        '''
        Add lines do the original image.

        Parameters:
            img (np.ndarray)
            r_points (list)
            l_points (list)
        Returns:
            img (np.ndarray)

        '''        
        r1 = r_points[0]
        r2 = r_points[1]
        l1 = l_points[0]
        l2 = l_points[1]
        red = (0, 0, 255)
        cv.circle(img, r1, 5, red, -1)  # Draw a red circle at r1
        cv.circle(img, r2, 5, red, -1)  # Draw a red circle at r2
        cv.circle(img, l1, 5, red, -1)  # Draw a red circle at l1
        cv.circle(img, l2, 5, red, -1)

        # draw lines
        blue = (255, 0, 0)
        cv.line(img,r1,r2,blue,2)
        cv.line(img,l1,l2,blue,2)

        return img
        

detlane()