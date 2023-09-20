import cv2
import numpy as np
from tqdm import tqdm
import os

def aggregate_cost_volume(cost_volume):
    height, width, max_disparity = cost_volume.shape
    aggregated_costs = np.zeros_like(cost_volume, dtype=np.float32)

    forward_pass = [(1, 0), (1, 1), (0, 1), (-1, 1)]  # Dy, Dx pairs for forward pass
    backward_pass = [(0, -1), (-1, -1), (-1, 0), (-1, 1)]  # Dy, Dx pairs for backward pass

    # Forward Pass
    for (dy, dx) in forward_pass:
        # Shift the cost volume by (dy, dx) and pad the shifted values
        shifted_costs = np.roll(cost_volume, (dy, dx), axis=(0, 1))
        shifted_costs[:dy, :] = np.inf
        shifted_costs[:, :dx] = np.inf
        aggregated_costs += shifted_costs

    # Backward Pass
    for (dy, dx) in backward_pass:
        # Shift the cost volume by (dy, dx) and pad the shifted values
        shifted_costs = np.roll(cost_volume, (dy, dx), axis=(0, 1))
        shifted_costs[dy:, :] = np.inf
        shifted_costs[:, dx:] = np.inf
        aggregated_costs += shifted_costs

    return np.sum(aggregated_costs, axis=2)  # Sum along axis=3

def warp_image(image, disparity_map):
    # TODO: Implement image warping
    # You can use cv2.warpPerspective or any other method for image warping
    # Make sure to handle borders properly

    # Placeholder code, replace with actual image warping
    warped_image = image  # Placeholder, replace this

    return warped_image

def Birchfield_Tomasi_dissimilarity(left_image, right_image, d):
    height, width = left_image.shape
    max_disparity = d

    left_cost_volume = np.full((height, width, max_disparity), np.inf, dtype=np.float32)
    right_cost_volume = np.full((height, width, max_disparity), np.inf, dtype=np.float32)

    for disparity in range(max_disparity):
        if disparity == 0:
            # Handle disparity 0 separately
            cost = np.abs(left_image - right_image)
            left_cost_volume[:, :, disparity] = cost
            right_cost_volume[:, :, disparity] = cost
        else:
            # Calculate the dissimilarity cost
            cost_left = np.abs(left_image[:, disparity:] - right_image[:, :-disparity])
            cost_right = np.abs(right_image[:, disparity:] - left_image[:, :-disparity])

            left_cost_volume[:, disparity:, disparity] = cost_left
            right_cost_volume[:, :-disparity, disparity] = cost_right

    # Find the disparity with the minimum cost
    left_disparity = np.argmin(left_cost_volume, axis=2)
    right_disparity = np.argmin(right_cost_volume, axis=2)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity


def semi_global_matching(left_image, right_image, d):
    # Perform Birchfield-Tomasi dissimilarity calculation
    left_cost_volume, right_cost_volume, left_disparity, right_disparity = Birchfield_Tomasi_dissimilarity(left_image, right_image, d)

    # Aggregate cost volume
    cost_volume = np.concatenate((left_cost_volume, right_cost_volume), axis=2)
    aggregated_cost_volume = aggregate_cost_volume(cost_volume)
    
    # Get the left and right aggregated disparities
    aggregated_disparity = aggregated_cost_volume.argmin(axis=0)

    # TODO: Save cost disparity and Semi-Global Matching disparity

    return aggregated_disparity


if __name__ == "__main__":
    img_list = os.listdir('./input')  # List of images
    ground_truth = None  # Ground truth disparity map

    d = 24  # Maximum disparity

    for i in range(len(img_list) - 1):
        # Perform Semi-Global Matching
        left_img = cv2.imread("./input/" + img_list[i], cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread("./input/" + img_list[i+1], cv2.IMREAD_GRAYSCALE)
        aggregated_disparity = semi_global_matching(left_img, right_img, d)

    warped_image_list = []  # List to store warped images
    for i, image in enumerate(img_list):
        # Warp image
        warped_image = warp_image(image, aggregated_disparity)
        warped_image_list.append(warped_image)

    boundary_range = d  # Boundary range for cropping
    cropped_ground_truth = ground_truth[boundary_range:-boundary_range, boundary_range:-boundary_range]

    # TODO: Aggregate warped images

    # TODO: Compute MSE and PSNR
    mse = np.inf
    print("mse: {mse}".format(mse=mse))

    psnr = np.inf
    print("psnr: {psnr}".format(psnr=psnr))

    # TODO: Save aggregated disparity