import cv2
import numpy as np
from tqdm import tqdm
import os

from Birchfield_Tomasi_dissimilarity import Sum_Of_Absolute_Differences
from aggregate_cost_volume import aggregate_cost_volume
from warp import warp_image


# Modify any parameters or any function itself if necessary.
# Add comments to parts related to scoring criteria to get graded.

def semi_global_matching(left_image, right_image, d):
    # TODO: Implement Semi-Global Matching

    #raise NotImplementedError("Semi_Global_Matching function has not been implemented yet")

    left_cost_volume, right_cost_volume, left_disparity, right_disparity = Sum_Of_Absolute_Differences(left_image, right_image, d)

    # TODO: save cost disparity

    cost_volume = left_cost_volume
    aggregated_cost_volume = aggregate_cost_volume(cost_volume)
    print(aggregated_cost_volume.shape)
    aggregated_disparity = aggregated_cost_volume.argmin(axis=2)
    print(aggregated_disparity)

    # TODO: save Semi Global Matching disparity


if __name__ == "__main__":
    img_list = os.listdir('./input')
    ground_truth = None
    # TODO: Load required images

    d = 24

    left_img = cv2.imread("./input/01_noise25.png", cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread("./input/02_noise25.png", cv2.IMREAD_GRAYSCALE)
    semi_global_matching(left_img, right_img, d)

    for i in range(len(img_list)):
        # TODO: Perform Semi-Global Matching
        break
    '''
    warped_image_list = list()
    for i, image in enumerate(img_list):
        # TODO: Warp image
        pass

    boundary_range = d
    cropped_ground_truth = ground_truth[boundary_range:-boundary_range, boundary_range:-boundary_range]

    # TODO: Aggregate warped images

    # TODO: Compute MSE and PSNR
    mse = np.inf
    print("mse: {mse}".format(mse=mse))

    psnr = np.inf
    print("psnr: {psnr}".format(psnr=psnr))

    # TODO: Save aggregated disparity
    '''
    None