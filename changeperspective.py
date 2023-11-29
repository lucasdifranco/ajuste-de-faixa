import cv2 as cv
import numpy as np
class changeperspective:

    def __init__(self,path,points):
        '''
        
        '''
        self.img_path = path
        self.src_points = points

        self.src_image = cv.imread(self.img_path)

    def changeperspective(self):

        dst_points = self.get_dst_points()
        dst_image = self.transform_img(dst_points,self.src_points)

        return dst_image
    
    def get_dst_points(self):
        '''
        Reads source image and get image points.
        '''
        width, height, channels = self.src_image.shape

        dst_points = np.array([ (0, 0),
                                (width, 0),
                                (width, height),
                                (0, height)], dtype=np.float32)
        
        return dst_points
    
    def transform_img(self,dst_points,src_points):
        '''
        
        '''

        # get pertspective matrix
        M = cv.getPerspectiveTransform(src_points, dst_points)

        transformed_img = cv.warpPerspective(self.src_image,M,(1200,1200))

        transformed_img = cv.resize(transformed_img,(350,400))

        return transformed_img