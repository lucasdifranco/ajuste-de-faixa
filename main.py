from lanedetection import detlane
from changeperspective import changeperspective
import numpy as np
import cv2
import os
def display_combined_images(img_path):
    '''
    Uses both 'changeperspective' and 'lanedetector' modules to detect lane lines and transform the image perspective.
    This function was designed to support the identification of defects using AI.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        None: This function modifies the image or performs transformations in place.

    Obs.: You can change to return both images, as separated or combined
    '''

    # Call detlane and changeperspective functions or methods to obtain images and points
    lane_detector = detlane(img_path)
    parameters = lane_detector.detlane()
    img, rpoints, lpoints = parameters

    src_points = np.array([rpoints[0], lpoints[0], lpoints[1], rpoints[1]], dtype=np.float32)

    transform_image = changeperspective(img_path, src_points)
    trans_img = transform_image.changeperspective()

    # Resize images to have the same height
    height = max(img.shape[0], trans_img.shape[0])
    width_img = img.shape[1]
    width_trans_img = trans_img.shape[1]

    # Resize the images to the same height
    resized_img = cv2.resize(img, (int(width_img * height / img.shape[0]), height))
    resized_trans_img = cv2.resize(trans_img, (int(width_trans_img * height / trans_img.shape[0]), height))

    # Concatenate the resized images horizontally
    combined_image = cv2.hconcat([resized_img, resized_trans_img])

    # Display the combined image
    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #return img, trans_img, combined_image

main_path = r"E:\2023.35 - VIABAHIA\Video\BA526-C-1-0-9\CAM 2"

for file in os.listdir(main_path):
    if file.endswith(".jpg"):
        file_path = os.path.join(main_path,file)
        print(file_path)
        display_combined_images(file_path)0
